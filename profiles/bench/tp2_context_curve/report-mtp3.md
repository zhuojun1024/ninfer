# TP-2 上下文档位曲线(原始启动脚本配置:K=3 MTP + vision)

本地测量,2026-09-20。同目录:`mtp3.json`(全部记录)、`mtp3_summary.csv`(汇总)、`mtp0.json`/报告(对照基线)。

## 与基线报告的口径差别

这份跑的是**你 `C:/Users/zhuojun/start-ninfer-tp2.ps1` 的默认配方**,即生产配置;
`mtp0.json` 那份是关闭投机的基线(`--no-prefix-reuse`、无 vision)。两者唯一的方法差异是
本配置**开启了前缀复用**(`context cache` 非 root-only),prefill 期间会写 host checkpoint,
因此小档位更容易出现模板前导(约 49 token)的缓存命中。

## 运行配方(原始脚本默认值)

| 项 | 值 |
|---|---|
| 硬件 | 2x RTX 5060 Ti 16 GiB(`--devices 0,1`,TP-2) |
| artifact | `D:/LLM/qwen3_8_27b_w4a4_w8a8.ninfer` |
| KV | fp8,capacity 204,800 token |
| 投机解码 | `--spec mtp --draft-tokens 3 --lm-head-draft`(K=3) |
| 视觉塔 | 开启(shard 1 arena 826.5 MiB) |
| 采样/思考 | `--temperature 0.7 --top-k 20 --top-p 0.80 --reasoning-effort medium --preserve-thinking` |
| 前缀复用 | 开启(`--host-state-slots 32`) |
| 服务 | 直接运行 `start-ninfer-tp2.ps1`(端口 3456);engine ready 31.9 s |
| 驱动 | `pwsh -File tools/win_port/context_curve.ps1 -BaseUrl http://127.0.0.1:3456/v1 -Label tp2-mtp3 -OutFile profiles/bench/tp2_context_curve/mtp3.json -Sizes '128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,196608' -Repetitions 3 -DecodeTokens 256` |

方法同 `docs/performance/methodology.md` 的单请求法:prefill = `prompt_n/prompt_ms`,
decode = `(predicted_n-1)/predicted_ms`,解码预算 256 tok(全部打满),横轴为 `usage.prompt_tokens`。

## 结果(K=3)

| context tokens | prefill tok/s | sd | reps | 平均缓存 tok | decode tok/s | sd | ms/token | prefill wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 118 | 787.9 | 0.0 | 1 | 33 | 71.03 | 2.95 | 14.08 | 0.15 |
| 268 | *1417.2 | 54.2 | 3 | 49 | 93.62 | 3.86 | 10.68 | 0.15 |
| 518 | *1542.6 | 32.3 | 3 | 49 | 89.94 | 3.81 | 11.12 | 0.30 |
| 1,018 | *1653.2 | 18.4 | 3 | 49 | 78.60 | 5.77 | 12.72 | 0.59 |
| 2,043 | *1645.2 | 9.3 | 3 | 49 | 75.49 | 6.37 | 13.25 | 1.21 |
| 4,093 | 1613.5 | 0.0 | 1 | 33 | 73.62 | 0.92 | 13.58 | 2.54 |
| 8,193 | 1608.5 | 0.5 | 3 | 0 | 72.85 | 3.03 | 13.73 | 5.09 |
| 16,393 | 1575.1 | 0.8 | 3 | 0 | 71.59 | 7.08 | 13.97 | 10.41 |
| 32,768 | 1495.9 | 0.6 | 3 | 0 | 66.04 | 4.55 | 15.14 | 21.90 |
| 65,543 | 1338.7 | 1.5 | 3 | 0 | 56.32 | 1.74 | 17.75 | 48.96 |
| 131,068 | 1111.2 | 0.5 | 3 | 0 | 54.91 | 6.95 | 18.21 | 117.96 |
| 196,618 | 946.7 | 0.1 | 3 | 0 | 53.68 | 2.85 | 18.63 | 207.69 |

\* = 该档 3 次重复全部命中缓存,prefill 一列只能用"含缓存"的平均值(相对全量 prefill 偏高 4-12%)。
其余档位取 `cache_n == 0` 的记录;≥8,193 全部 3 次无缓存。

## 与 MTP0 基线对比(prefill 用两边的"全记录"均值,口径一致)

| context tokens | MTP0 prefill | MTP3 prefill | Δ | MTP0 decode | MTP3 decode | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 118 | 891.1 | 862.2 | -3.2% | 31.09 | 71.03 ± 2.95 | **2.28x** |
| 268 | 1392.4 | 1417.2 | +1.8% | 31.11 | 93.62 ± 3.86 | **3.01x** |
| 518 | 1518.0 | 1542.6 | +1.6% | 31.17 | 89.94 ± 3.81 | **2.89x** |
| 1,018 | 1637.4 | 1653.2 | +1.0% | 31.09 | 78.60 ± 5.77 | **2.53x** |
| 2,043 | 1647.2 | 1645.2 | -0.1% | 30.99 | 75.49 ± 6.37 | **2.44x** |
| 4,093 | 1665.0 | 1633.1 | -1.9% | 30.91 | 73.62 ± 0.92 | **2.38x** |
| 8,193 | 1643.3 | 1608.5 | -2.1% | 30.71 | 72.85 ± 3.03 | **2.37x** |
| 16,393 | 1595.1 | 1575.1 | -1.3% | 30.42 | 71.59 ± 7.08 | **2.35x** |
| 32,768 | 1510.1 | 1495.9 | -0.9% | 29.78 | 66.04 ± 4.55 | **2.22x** |
| 65,543 | 1349.1 | 1338.7 | -0.8% | 28.64 | 56.32 ± 1.74 | **1.97x** |
| 131,068 | 1117.9 | 1111.2 | -0.6% | 26.65 | 54.91 ± 6.95 | **2.06x** |
| 196,618 | 951.4 | 946.7 | -0.5% | 24.98 | 53.68 ± 2.85 | **2.15x** |

## 观察

- **decode**:K=3 相对 MTP0 加速 2.15x(200K)到 3.01x
  (峰值在 268 token);全程 53.7-93.6 tok/s,
  200K 仍有 53.7 tok/s。低上下文段(268 token)是投机收益最高点,
  之后随上下文增长,verify 成本上升,加速比回落到 ~2x。
- **投机方差显著大于 MTP0**:同一档 3 次重复的 decode sd 最大 7.08 tok/s
  (MTP0 最大仅 0.04)。这是投机解码的固有性质 —— 每轮接受数取决于内容,不是测量噪声;
  单点比较应看均值和区间,不要拿单次数字。
- **prefill 与 MTP0 几乎一致但系统性低 0.5-2%**(大档位):投机不影响 prefill,
  这点差异来自本配置开启前缀复用后 prefill 期间的 host checkpoint 写入(32 槽 × 73.4 MiB/卡 pinned),
  以及常驻的 vision arena。
- **小档位(<4K)由固定开销 + 模板缓存主导**:118 token 时 prefill 只有 ~790-860 tok/s,
  到 4K 才到平台 ~1.65k tok/s。

## 已知缺口:这条路线不暴露投机接受率

响应 `timings` 块里没有 `draft_n`/`draft_n_accepted`(表里 accept% 因此为空),日志也只有
prefill/decode 两相。原因在源码:`src/serve/openai_chat_response.cpp:247` 走
`outcome_timings()`,它读取的 `result.speculative.*` 由单卡路线(`engine_core.h:864`)填充,
而 `src/runtime/engine/tp2_generation_core.cpp` 从不写 `result.speculative`(它只在内部
`timing.committed += decision.accepted_tokens` 里统计,供 `NINFER_TP2_TIMING=1` 的每轮调试输出)。
所以 `docs/performance/methodology.md` 定义的 **Spec acceptance / tokens-per-round 在 TP-2 服务路线上目前拿不到**。
两种补法:① 让 TP-2 core 回填 `result.speculative`(推荐,单卡路线已有同一结构体);
② 临时用 `NINFER_TP2_TIMING=1` 重启并解析每轮日志(粗糙,需重启服务)。

## 局限

- 仅 C=1 单请求;并发曲线另测。
- `* ` 标记的 4 个档位 prefill 含缓存,不能与 MTP0 的清洁点直接比。
- 未测 DFlash/DFlash2,也未测视觉输入。
