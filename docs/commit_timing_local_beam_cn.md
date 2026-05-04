# Commit-Timing-Aware Local Beam Search 设计说明

## 方法摘要

Commit-Timing-Aware Local Beam Search 是一个 train-free、推理时、baseline-preserving 的扩散代码解码方法。它把一类结构性失败解释为 commit timing error：模型过早提交 `def`、`return`、冒号、括号、缩进、比较符、调用边界等结构锚点后，后续 denoising 会在错误骨架里继续细化，形成 stable-but-wrong trajectory。

方法只使用正常 denoising 已经产生的 logits，以及相邻 diffusion step 的 logits。它不做 perturbation-KL，不为每个候选 token 额外重跑 counterfactual forward。高风险 token 只触发一次小型局部 beam，默认 beam size 4、horizon 2，并始终保留 baseline particle。

## 核心流程

1. 对仍 masked 或未不可逆提交的位置计算 top-k 归一化 entropy。
2. 对同一位置计算相邻 step temporal KL，并用当前有效 masked 位置内的 rank percentile 归一化。
3. 根据 top-K token 文本和邻近已提交上下文计算结构敏感分数。
4. 用 `Risk = H_norm * KL_norm * (1 + alpha_struct * S_struct)` 排序。
5. 默认只取 top-1 高风险位置，且每个样本最多触发一次 local beam。
6. beam 粒子包含 baseline、若干 top token candidate、delay particle。
7. 粒子只短程 rollout 2 到 3 step，评分使用模型概率、稳定性下降、结构检查和坏模式惩罚。
8. 非 baseline 粒子必须同时超过 baseline margin 和 branch margin，且不能造成解析质量明显退化。
9. 如果证据不清楚，保持 baseline 或 delay，不做硬替换。

## 伪代码

```text
prev_logits = None
branch_events = 0

for diffusion step t:
    logits = model(x_t)
    valid = masked_or_uncommitted_positions(x_t)

    if local_beam_enabled and prev_logits is not None:
        for i in valid:
            H[i] = topk_entropy(logits[i])
            KL[i] = topk_temporal_kl(logits[i], prev_logits[i])
            S[i] = structure_score(topK(logits[i]), decoded_neighbors(i))
            Risk[i] = H[i] * KL_norm[i] * (1 + alpha * S[i])

        i* = top risky position satisfying tau_H, tau_KL, tau_R

        if i* exists and branch_events < max_events:
            particles = [
                baseline(argmax token),
                candidate(top-2/top-3 tokens),
                delay(mask remains unresolved)
            ]

            rollout each particle for H short diffusion steps
            score = model_score + stability_score + structure_score - badness

            if best_nonbaseline clears conservative margins:
                commit/select it
            elif delay is configured and useful:
                delay this position
            else:
                keep baseline

            branch_events += 1

    continue normal diffusion transfer/commit policy
    prev_logits = logits
```

## 主要公式

归一化 entropy 使用 top-k 近似：

```text
H_topk(i,t) = - sum_v p_k(v) log(p_k(v) + eps)
H_norm(i,t) = H_topk(i,t) / log(k)
```

temporal KL 使用当前和前一步 top-k union：

```text
KL(i,t) = sum_v p_t(v) log((p_t(v)+eps)/(p_{t-1}(v)+eps))
KL_norm(i,t) = rank_percentile(KL(i,t))
```

结构加权：

```text
W_struct(i,t) = 1 + alpha_struct * S_struct(i,t)
Risk(i,t) = H_norm(i,t) * KL_norm(i,t) * W_struct(i,t)
```

局部 beam 粒子评分：

```text
ParticleScore =
  ModelScore
+ 0.5 * StabilityScore
+ 0.5 * StructureScore
- BadnessPenalty
```

其中 `StabilityScore` 近似为触发点 entropy 降低量减去 KL 惩罚，`StructureScore` 来自 format、括号、AST parse 等最小结构检查。

## 配置参数

核心 CLI flags：

```bash
--local_beam_enabled
--local_beam_mode entropy_only|kl_only|entropy_kl|entropy_kl_struct|beam|delay_only
--local_beam_size 4
--local_beam_top_k 5
--local_beam_horizon 2
--local_beam_max_events 1
--local_beam_tau_entropy 0.45
--local_beam_tau_kl 0.8
--local_beam_tau_risk 0.45
--local_beam_lambda_kl 1.0
--local_beam_struct_weight 0.75
--local_beam_margin_base 0.15
--local_beam_margin_branch 0.05
--local_beam_preserve_baseline
--eval_compare_baseline
```

消融 preset：

```bash
--preset lb_entropy_only
--preset lb_kl_only
--preset lb_entropy_kl
--preset lb_entropy_kl_struct
--preset lb_delay_only
--preset local_beam
```

## 评测协议

主对比仍是 baseline decoding。报告：

- pass rate / parse rate / format rate
- latency 和 `latency_ratio_vs_baseline`
- `avg_extra_forwards`
- `avg_local_beam_branch_events`
- `avg_local_beam_beam_size`
- `total_local_beam_accepted_alternatives`
- `total_local_beam_delay_count`

若开启 `--eval_compare_baseline`，每个样本额外运行匹配 baseline，并报告：

- `recovered_baseline_failures`
- `damaged_baseline_successes`
- `net_change`
- `parse_recovered_baseline_failures`
- `parse_damaged_baseline_successes`
- `parse_net_change`

正式 fair comparison 不使用隐藏测试做解码期选择。`--eval_compare_baseline` 只用于离线统计 damage/recovery。

## 与已有方法的区别

EB-Sampler：只根据 entropy 决定是否 unmask，缺少 temporal stability 和结构敏感性。本方法要求 entropy、temporal KL 和结构权重共同触发。

strict PF：仍然容易在单点高风险时直接改变当前 trajectory。本方法把高风险解释为“需要保留短程假设”，baseline 粒子始终存在。

PF/RDD：RDD 会 rollback/remask 已走坏的单一路径，容易过度干预。本方法默认不回滚，不扰动唯一轨迹，而是在 commit 前做短程局部选择。

SWD：若指简单结构加权 decoding，它通常是单轨迹打分。本方法的关键不是静态权重，而是 temporal KL 触发的短程多假设 commit timing。

普通 AR beam search：AR beam 是从左到右全局展开；本方法只在 diffusion denoising 中少量结构风险点局部展开，且复用正常 step logits，不改变模型训练方式。
