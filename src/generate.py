"""vLLM 批量生成：对评测集批量生成回复。"""
# src/generate.py
import argparse, json
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
p.add_argument("--adapter", default=None)      # LoRA 目录，可选
p.add_argument("--split", default="test")
p.add_argument("--n", type=int, default=1)     # 每个 prompt 采样几条
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--max_tokens", type=int, default=512)
p.add_argument("--out", required=True)
args = p.parse_args()

ds = load_from_disk("data/support")[args.split]
llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048, seed=0,
          enable_lora=args.adapter is not None, max_lora_rank=64)
sp = SamplingParams(n=args.n, temperature=args.temperature,
                    top_p=0.95 if args.temperature > 0 else 1.0, max_tokens=args.max_tokens)
lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None

outputs = llm.chat([ex["prompt"] for ex in ds], sp, lora_request=lora)
with open(args.out, "w") as f:
    for ex, out in zip(ds, outputs):
        f.write(json.dumps({
            "prompt": ex["prompt"],
            "question": ex["prompt"][-1]["content"],
            "reference": ex["completion"][0]["content"],
            "intent": ex["intent"],
            "responses": [o.text for o in out.outputs],
        }, ensure_ascii=False) + "\n")