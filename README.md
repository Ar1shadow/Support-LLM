# support-llm

中文 | [English](README_en.md)

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
  tests/             # judge.py 的单元测试（pytest，不调 API）
  docs/              # 计划与报告
  README.md
  README_en.md
```

## 环境

需要 Python ≥ 3.10。

## 数据分析

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


## SFT

模型 Qwen/Qwen3-4B-Instruct-2507，LoRA r=16，全部线性层，lr 2e-4，2 epoch，有效 batch 16，fp16，2 张 GPU。用时 5853 s，最佳 eval loss 0.555（checkpoint-1000）。

评测集：test 270 条（Bitext，同分布），OOD 50 条。pointwise 指 `eval_responses.py` 的 1 到 10 分打分（带参考答案），pairwise 指 `winrate.py` 的两两对比（不带参考答案，交换顺序各判一次）。

### 裁判 prompt 的改动

v1（e849d26）的 pointwise 有四条评分标准和一条硬规则：直接写出客户没给过的具体值（电话、URL、时效、费用等）最高 4 分。pairwise 只有一句 "judge helpfulness, correctness and whether it invents specific facts"，没有评分标准。两个裁判结论相反：pointwise 判 SFT 更好，pairwise 判 base 赢 70%。逐条对比有 80 条冲突。

v3（7b75826）让两个裁判共用同一套标准：

- 共用 `CRITERIA`（四条评分标准）和 `INVENTED_RULES`（编造的定义）。
- `{{...}}` 占位符视为对系统真实数据的引用，系统发送时填值，不算编造。
- 编造分两类。value 是客户没给、也没用占位符的具体值，加 "typically" 之类的限定词也算。claim 是用纯文字写出、不带占位符的店铺规则、政策、能力或已完成的动作，例如 "orders can only be cancelled before shipping"、"I have cancelled your order"。
- 通用步骤（"log in, open My Orders, choose Cancel"）和不含具体值的条件句不算编造。
- pointwise 返回编造的原文和类型。pairwise 只返回每个回复有哪类编造，不引用原文，原文可在结果文件的 `items` 里自己查。
- 只有一方编造时，编造的一方判负，除非另一方没有回答问题。两方都编造时，选编造影响较小的一方。
- `max_tokens` 从 1000 提到 4000。thinking 的推理 token 也计入上限，1000 时大量输出被截断。


#### 结果

pointwise（均分，括号内是裁判判定的编造率 value / claim）：

| | base test | SFT test | base OOD | SFT OOD |
|---|---|---|---|---|
| v1 均分 | 6.07 | 7.19 | 6.00 | 5.98 |
| v3 均分 | 5.73 | 6.48 | 5.12 | 5.30 |
| v3 编造率 | 29.6% / 19.6% | 5.2% / 18.9% | 20% / 36% | 2% / 24% |
| 正则 invented_rate | 15.9% | 0 | 6% | 0 |

v3 下 SFT 减 base 的配对差：test +0.75（95% CI 0.37 到 1.14），OOD +0.18（95% CI -0.76 到 1.14）。

pairwise（SFT 胜 / base 胜 / 平，mean_score 及 95% CI）：

| | test | OOD |
|---|---|---|
| v1 | 13.4% / 70.3% / 16.4%，-0.57 | 14% / 80% / 6%，-0.66 |
| v3 | 32.2% / 53.7% / 14.1%，-0.22 [-0.32, -0.11] | 22% / 66% / 12%，-0.43 [-0.66, -0.18] |

v3 下两个裁判的冲突从 80 条降到 24 条。

>- SFT 直接写出具体值的编造从约 30% 降到约 5%（test）。纯文字的 claim 没有减少，test 上两边都在 19% 左右。SFT 的 claim 集中在 `delivery_options` 和 `check_refund_policy`，这两类的 Bitext 参考答案本身就用纯文字写配送选项和退款政策。
>- OOD 上 SFT 没有提升。
>- test 上两边都没编造的 116 条里，base 赢 88 条，SFT 赢 18 条。SFT 常常只追问细节，不给步骤。
>- test 里有 13 条问题带客户自己给的编号（如 bill #00108），SFT 全部改写成占位符，base 都保留了。Bitext 的问题里编号本来就是占位符，SFT 学会了这个习惯，部署时会出错。
>- 下一步：训练数据的问题里保留真实编号，并挑选或补充直接给出步骤的回复。

### 已知问题

claim 的判定噪声大。对 SFT test，pairwise 判出 92 条有 claim，pointwise 判出 51 条，重合 45 条。"Live Chat on our website" 这类说法两边判法不一致。人工校准还没做。

结果文件：v1 为 `results/eval_{base,sft}_{test,ood}.json` 和 `results/winrate_gen_*.json`，v3 为 `results/eval_*_v3.json` 和 `results/winrate_v3_*.json`。

## 消融实验（LoRA 配置）

以 SFT 的 r16 为基线，每次只改一项 LoRA 配置：rank 4、rank 64、只训注意力层、QLoRA。其余设置与 SFT 相同（alpha = 2r，seed 42），评测用 v3 pointwise。各组生成质量在 test 上没有显著差异，差别只在训练开销。

### 结果

| 实验 | 可训练参数 | 每卡峰值显存 (GB) | 训练时间 (min) | eval loss |
|---|---|---|---|---|
| r4 | 8.26M | 10.59 | 104.5 | 0.575 |
| r16（基线） | 33.03M | 10.99 | 97.6 | 0.555 |
| r64 | 132.12M | 12.60 | 105.5 | 0.545 |
| attn | 11.80M | 10.18 | 94.5 | 0.591 |
| qlora | 33.03M | 5.22 | 111.4 | 0.560 |

差值是与 r16 逐条配对相减的均值及 95% CI，编造率为 value / claim：

| 实验 | test 均分 | test 差值 | test 编造率 | OOD 均分 | OOD 差值 | OOD 编造率 |
|---|---|---|---|---|---|---|
| r4 | 6.33 | -0.15 [-0.39, 0.09] | 6.7% / 20.0% | 5.34 | +0.04 [-0.34, 0.40] | 2% / 24% |
| r16 | 6.48 | | 5.2% / 18.9% | 5.30 | | 2% / 24% |
| r64 | 6.29 | -0.20 [-0.47, 0.07] | 5.2% / 21.1% | 4.64 | -0.66 [-1.24, -0.10] | 0% / 40% |
| attn | 6.33 | -0.16 [-0.42, 0.10] | 6.7% / 19.6% | 5.14 | -0.16 [-0.60, 0.26] | 4% / 28% |
| qlora | 6.65 | +0.17 [-0.14, 0.46] | 5.9% / 17.8% | 5.12 | -0.18 [-0.86, 0.48] | 2% / 32% |

### 结论

- 各组学到的是同一套 Bitext 模板。test 上与 r16 的差值 CI 都跨过 0。SFT 发现的问题也都还在，比如 13 条带客户编号的问题，五组全部把编号改成了占位符。瓶颈在数据，不在 LoRA 配置。
- rank 的收益递减，r64 多用 15% 显存。只训注意力层的 eval loss 最差。QLoRA 显存减半，质量不变，训练慢 14%（4-bit 反量化）。默认用 r16，显存不够时用 QLoRA。

### 问题与局限

- 新版 TRL 会把量化模型的 adapter 转成 bf16，和 fp16 的 GradScaler 冲突，QLoRA 训练报错。修复（2f33d0f）在训练前把 adapter 转回 fp32。
- 每组只跑了一个 seed，训练时间有约 ±7% 的波动，小差异不可靠。r64 在 OOD 上低 0.66，但 OOD 只有 50 条，没有追查。
