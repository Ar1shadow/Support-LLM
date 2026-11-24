"""SFT 训练 TRL + PEFT LoRA"""
# src/train_sft.py
# Single GPU : python src/train_sft.py --run_name xxx
# Multi GPU  : accelerate launch --num_processes 2 src/train_sft.py --run_name xxx --grad_accum 1
#              (DDP: every GPU holds a full model replica and processes a different slice of the batch)
import argparse, json, os, torch
from accelerate import PartialState
from datasets import load_from_disk
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
p.add_argument("--run_name", required=True)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=None)          # 默认取 2 * rank
p.add_argument("--targets", choices=["all", "attn"], default="all")
p.add_argument("--qlora", action="store_true")
p.add_argument("--dora", action="store_true")
p.add_argument("--n_train", type=int, default=None)        # 只用前 n 条，消融和过拟合测试用
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--bs", type=int, default=8)                # per-GPU micro batch size
p.add_argument("--grad_accum", type=int, default=2)        # keep bs * grad_accum * n_gpu fixed across runs
# bfloat16 needs Ampere or newer (A100, L4, RTX 30xx+); use float16 on T4 / P100 (Kaggle).
p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
args = p.parse_args()
torch_dtype = getattr(torch, args.dtype)   # weight dtype and mixed-precision compute dtype

state = PartialState()  # process info: rank, number of processes (GPUs) in DDP

# Dataset columns (built by prepare_data.py), conversational prompt-completion format:
#   prompt:     [{"role": "system", ...}, {"role": "user", ...}]
#   completion: [{"role": "assistant", ...}]
ds = load_from_disk("data/support")
train = ds["sft"].select_columns(["prompt", "completion"])   # 去掉 intent 等多余列
if args.n_train:
    train = train.select(range(args.n_train))
val = ds["val"].select_columns(["prompt", "completion"]) # validation set

targets = "all-linear" if args.targets == "all" else ["q_proj", "k_proj", "v_proj", "o_proj"]  # the update obj layer
peft_config = LoraConfig(
    r=args.rank, lora_alpha=args.alpha or 2 * args.rank, lora_dropout=0.05,
    target_modules=targets, use_dora=args.dora, task_type="CAUSAL_LM")

model_kwargs = {"dtype": torch_dtype}
if args.qlora:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch_dtype)
    # A 4-bit model cannot be moved after loading, so place each DDP process's
    # replica on its own GPU explicitly (avoids device_map="auto" sharding).
    model_kwargs["device_map"] = {"": state.local_process_index}

cfg = SFTConfig(
    output_dir=f"runs/{args.run_name}", run_name=args.run_name, report_to="wandb",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
    gradient_accumulation_steps=args.grad_accum,     # 有效 batch = bs * grad_accum * N_GPU = 16
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=20,
    # max_length counts tokens of the whole chat-templated sequence(system + user + assistant + special tokens)
    max_length=1024,
    # Mixed precision follows --dtype; fp16 also enables loss scaling to avoid gradient underflow.
    bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16",
    # Loss only on the assistant completion; prompt tokens are masked with -100.
    completion_only_loss=True,
    gradient_checkpointing=True,
    # Non-reentrant checkpointing is required for DDP + LoRA (frozen base weights),
    # otherwise DDP may raise "Expected to mark a variable ready only once".
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    logging_steps=10, eval_strategy="steps", eval_steps=100,
    # save_steps must be a multiple of eval_steps for load_best_model_at_end.
    save_strategy="steps", save_steps=200, save_total_limit=2,
    # Reload the checkpoint with the lowest eval loss after training
    # (the best checkpoint is never deleted by save_total_limit).
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    model_init_kwargs=model_kwargs, seed=42,
)
trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=train,
                     eval_dataset=val, peft_config=peft_config)
if trainer.is_world_process_zero():
    trainer.model.print_trainable_parameters()

torch.cuda.reset_peak_memory_stats()
result = trainer.train()
# Read peak memory before the final evaluate() so it reflects training only.
# It covers the current process's GPU only; under DDP every GPU holds a full
# replica, so rank 0's value is representative of per-GPU memory.

peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9   # closer to nvidia-smi

trainer.save_model(f"adapters/{args.run_name}")    # 只存 adapter，几十 MB
# evaluate() is a collective op under DDP: it must run on every rank.
final_eval_loss = trainer.evaluate()["eval_loss"]  # eval loss of the best (reloaded) model

# for multiple GPU
if trainer.is_world_process_zero():
    stats = {**vars(args),
             "n_gpu": state.num_processes,
             "effective_batch": args.bs * args.grad_accum * state.num_processes,
             "train_runtime_s": result.metrics["train_runtime"],
             "train_loss": result.metrics["train_loss"],          # mean train loss over the run
             "peak_mem_gb_per_gpu": peak_alloc_gb,
             "peak_reserved_gb_per_gpu": peak_reserved_gb,
             "best_eval_loss": trainer.state.best_metric,
             "best_checkpoint": trainer.state.best_model_checkpoint,
             "final_eval_loss": final_eval_loss,
             # Every logged train/eval loss, so curves can be re-plotted without wandb.
             "log_history": trainer.state.log_history}
    os.makedirs("results", exist_ok=True)
    with open(f"results/train_{args.run_name}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print({k: v for k, v in stats.items() if k != "log_history"})
