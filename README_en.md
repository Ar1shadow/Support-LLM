# support-llm

[中文](README.md) | English

Post-training an LLM for customer support: SFT, then DPO, then (optionally) GRPO. Evaluation uses rule-based metrics and an LLM judge.

## Layout

```
support-llm/
  data/              # processed data (not committed), plus ood_en.jsonl / ood_fr.jsonl
  src/
    prepare_data.py  # data split
    metrics.py       # rule metrics: invented details, placeholders, length
    judge.py         # LLM judge client
    generate.py      # batch generation with vLLM
    eval_responses.py
    winrate.py
    train_sft.py
    merge_lora.py
    build_prefs.py
    train_dpo.py
    train_grpo.py    # optional
  results/           # one json per evaluation run, committed
  tests/             # unit tests for judge.py (pytest, no API calls)
  docs/              # plans and reports
  README.md
  README_en.md
```

## Environment

Requires Python ≥ 3.10.

### Local machine (Mac, code only)

No torch. Install only what data processing, the tokenizer and the judge client need:

```bash
python3 -m venv .venv-local && source .venv-local/bin/activate
pip install -U datasets transformers openai numpy jinja2
```

### Cloud GPU (training, generation, evaluation)

Training and inference use separate virtual environments so that vLLM and the training stack do not pull conflicting torch versions.

```bash
# training
python -m venv .venv-train && source .venv-train/bin/activate
pip install -U torch transformers trl peft datasets accelerate bitsandbytes wandb openai
pip freeze > requirements-train.txt
deactivate

# generation, evaluation, serving
python -m venv .venv-infer && source .venv-infer/bin/activate
pip install -U vllm lm-eval datasets openai
pip freeze > requirements-infer.txt
deactivate
```


## SFT

Model Qwen/Qwen3-4B-Instruct-2507, LoRA r=16 on all linear layers, lr 2e-4, 2 epochs, effective batch 16, fp16, 2 GPUs. Training took 5853 s; best eval loss 0.555 (checkpoint-1000).

Eval sets: 270 test items (Bitext, in distribution) and 50 OOD items. "Pointwise" means the 1 to 10 score from `eval_responses.py`, which sees a reference reply. "Pairwise" means the A/B comparison from `winrate.py`, which sees no reference and judges each pair in both orders.

### Judge prompt changes

In v1 (e849d26) the pointwise prompt had four scoring criteria and one hard rule: writing out a concrete detail the customer never gave (phone, URL, processing time, fee and so on) caps the score at 4. The pairwise prompt had a single line, "judge helpfulness, correctness and whether it invents specific facts", and no criteria. The two judges disagreed: pointwise preferred SFT, while pairwise had base winning 70%. Item by item, 80 test items conflicted.

v3 (7b75826) gives both judges the same standard:

- Both prompts share `CRITERIA` (the four scoring criteria) and `INVENTED_RULES` (what counts as invented).
- A `{{...}}` placeholder is a reference to the store's real data that the system fills in at send time. It is never an invented fact.
- There are two kinds of invention. A value is a concrete detail the customer did not give, written out instead of as a placeholder; hedging such as "typically" does not excuse it. A claim is a store-specific rule, policy, capability or completed action stated in plain text with no placeholder, for example "orders can only be cancelled before shipping" or "I have cancelled your order".
- Generic steps ("log in, open My Orders, choose Cancel") and conditional statements with no concrete value are not invented.
- Pointwise returns each invented span with its kind. Pairwise returns only which kinds each reply contains; the raw replies are in the `items` field of the output file for manual checking.
- When only one reply invents, it loses unless the other reply fails to answer the question. When both invent, the judge prefers the one whose inventions matter less.
- `max_tokens` went from 1000 to 4000. Reasoning tokens count toward the limit, and at 1000 many outputs were cut off.


#### Results

Pointwise (mean score; invention rates are value / claim as flagged by the judge):

| | base test | SFT test | base OOD | SFT OOD |
|---|---|---|---|---|
| v1 mean | 6.07 | 7.19 | 6.00 | 5.98 |
| v3 mean | 5.73 | 6.48 | 5.12 | 5.30 |
| v3 invention rate | 29.6% / 19.6% | 5.2% / 18.9% | 20% / 36% | 2% / 24% |
| regex invented_rate | 15.9% | 0 | 6% | 0 |

Under v3 the paired difference, SFT minus base, is +0.75 on test (95% CI 0.37 to 1.14) and +0.18 on OOD (95% CI -0.76 to 1.14).

Pairwise (SFT wins / base wins / ties, then mean_score with 95% CI):

| | test | OOD |
|---|---|---|
| v1 | 13.4% / 70.3% / 16.4%, -0.57 | 14% / 80% / 6%, -0.66 |
| v3 | 32.2% / 53.7% / 14.1%, -0.22 [-0.32, -0.11] | 22% / 66% / 12%, -0.43 [-0.66, -0.18] |

Under v3 the number of items where the two judges conflict fell from 80 to 24.

>- On test, SFT cut written-out invented values from about 30% to about 5%. Plain-text claims did not drop; both models sit near 19% on test. SFT's claims cluster in `delivery_options` and `check_refund_policy`, where the Bitext reference answers themselves describe shipping options and refund policy in plain text.
>- SFT shows no gain on OOD.
>- Among the 116 test items where neither reply invents anything, base wins 88 and SFT wins 18. SFT often asks for more details instead of giving steps.
>- 13 test questions contain an ID the customer supplied (for example bill #00108). SFT replaced all 13 with a placeholder; base kept every one. Bitext questions already use placeholders for IDs, and SFT picked up the habit, which would break in deployment.
>- Next step: keep real IDs in the training questions, and select or add replies that give direct steps.

### Known issues

Claim detection is noisy. On SFT's test replies, pairwise flags 92 items with a claim and pointwise flags 51, with 45 in common. The two judges handle phrases like "Live Chat on our website" differently. Manual calibration has not been done yet.

Result files: v1 is `results/eval_{base,sft}_{test,ood}.json` and `results/winrate_gen_*.json`; v3 is `results/eval_*_v3.json` and `results/winrate_v3_*.json`.

## Ablation (LoRA configuration)

Using the SFT r16 run as the baseline, each run changes one LoRA setting: rank 4, rank 64, attention layers only, or QLoRA. Everything else matches SFT (alpha = 2r, seed 42), and evaluation uses v3 pointwise. Generation quality on test does not differ significantly between runs; only training cost differs.

### Results

| Run | Trainable params | Peak memory per GPU (GB) | Training time (min) | Eval loss |
|---|---|---|---|---|
| r4 | 8.26M | 10.59 | 104.5 | 0.575 |
| r16 (baseline) | 33.03M | 10.99 | 97.6 | 0.555 |
| r64 | 132.12M | 12.60 | 105.5 | 0.545 |
| attn | 11.80M | 10.18 | 94.5 | 0.591 |
| qlora | 33.03M | 5.22 | 111.4 | 0.560 |

The difference is the mean per-item paired difference against r16 with its 95% CI. Invention rates are value / claim:

| Run | Test mean | Test diff | Test invented | OOD mean | OOD diff | OOD invented |
|---|---|---|---|---|---|---|
| r4 | 6.33 | -0.15 [-0.39, 0.09] | 6.7% / 20.0% | 5.34 | +0.04 [-0.34, 0.40] | 2% / 24% |
| r16 | 6.48 | | 5.2% / 18.9% | 5.30 | | 2% / 24% |
| r64 | 6.29 | -0.20 [-0.47, 0.07] | 5.2% / 21.1% | 4.64 | -0.66 [-1.24, -0.10] | 0% / 40% |
| attn | 6.33 | -0.16 [-0.42, 0.10] | 6.7% / 19.6% | 5.14 | -0.16 [-0.60, 0.26] | 4% / 28% |
| qlora | 6.65 | +0.17 [-0.14, 0.46] | 5.9% / 17.8% | 5.12 | -0.18 [-0.86, 0.48] | 2% / 32% |

### Findings

- Every run learned the same Bitext template. On test, every CI against r16 includes 0. The problems found in SFT remain too: for the 13 questions with a customer's own number, all five runs replace the number with a placeholder. The bottleneck is the data, not the LoRA configuration.
- Rank has diminishing returns, and r64 uses 15% more memory. Attention-only has the worst eval loss. QLoRA halves memory with the same quality but trains 14% slower (4-bit dequantization). Default to r16; use QLoRA when memory is short.

### Problems and limitations

- Recent TRL casts a quantized model's adapter to bf16, which clashes with the fp16 GradScaler and crashed QLoRA training. The fix (2f33d0f) casts the adapter back to fp32 before training.
- One seed per run, and training times vary by about ±7%, so small gaps are unreliable. r64 scores 0.66 lower on OOD, but OOD has only 50 items and this was not followed up.
