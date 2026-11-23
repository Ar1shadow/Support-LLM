# support-llm

客服场景 LLM 后训练：SFT → DPO →（可选）GRPO，配套规则指标与裁判模型评测。

## 目录

```
support-llm/
  data/              # 处理后的数据（不提交），另放 ood_en.jsonl / ood_fr.jsonl
  src/
    prepare_data.py  # 切分数据
    metrics.py       # 规则指标：编造信息、占位符、长度
    judge.py         # 裁判模型客户端
    generate.py      # vLLM 批量生成
    eval_responses.py
    winrate.py
    train_sft.py
    merge_lora.py
    build_prefs.py
    train_dpo.py
    train_grpo.py    # 可选
  results/           # 每次评测的 json，提交
  README.md
```

## 环境

需要 Python ≥ 3.10。

### 本机（Mac，只写代码）

不装 torch，只装数据处理、tokenizer 和裁判客户端所需的包：

```bash
python3 -m venv .venv-local && source .venv-local/bin/activate
pip install -U datasets transformers openai numpy jinja2
```

### 云 GPU（训练、生成、评测）

训练与推理分两个虚拟环境，避免 vLLM 与训练栈的 torch 版本冲突。

```bash
# 训练环境
python -m venv .venv-train && source .venv-train/bin/activate
pip install -U torch transformers trl peft datasets accelerate bitsandbytes wandb openai
pip freeze > requirements-train.txt
deactivate

# 生成、评测、部署环境
python -m venv .venv-infer && source .venv-infer/bin/activate
pip install -U vllm lm-eval datasets openai
pip freeze > requirements-infer.txt
deactivate
```

已有 `requirements-*.txt` 时，直接 `pip install -r requirements-train.txt` 复现。

## 凭证

```bash
hf auth login     # Hugging Face token
wandb login
export JUDGE_BASE_URL=...  JUDGE_API_KEY=...  JUDGE_MODEL=...
```

`JUDGE_*` 变量可写进 `.env`（已在 `.gitignore` 中）。
