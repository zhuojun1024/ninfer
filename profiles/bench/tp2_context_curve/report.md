# TP-2 上下文档位曲线(prefill / decode,MTP0)

本地测量,2026-09-20。同目录数据:`mtp0.json`(全部记录)、`mtp0_summary.csv`(汇总)。

## 运行环境与配方

| 项 | 值 |
|---|---|
| 硬件 | 2x NVIDIA GeForce RTX 5060 Ti 16 GiB(TP-2,engine `--devices 0,1`) |
| artifact | `D:/LLM/qwen3_8_27b_w4a4_w8a8.ninfer`(Qwen3.8-27B,w4a4 + w8a8) |
| KV | fp8,capacity 204,800 token |
| 投机解码 | 关闭(MTP0,未传 `--spec`) |
| 视觉塔 | 关闭(未传 `--vision`,纯文本曲线) |
| 前缀复用 | `--no-prefix-reuse`(日志 `context cache | root only`) |
| 服务命令 | `ninfer-serve.exe <artifact> --devices 0,1 --max-context 204800 --port 3457 --host 127.0.0.1 --kv-dtype fp8 --log-level info --no-prefix-reuse` |
| 驱动命令 | `pwsh -File tools/win_port/context_curve.ps1 -BaseUrl http://127.0.0.1:3457/v1 -Label tp2-mtp0 -OutFile profiles/bench/tp2_context_curve/mtp0.json -Sizes '128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,196608' -Repetitions 3 -DecodeTokens 256` |

engine ready 29.7 s;加载后每卡空闲 1184 MiB(capacity 204,800、无 vision、无 host checkpoint 环)。

## 方法

单请求法(同 `docs/performance/methodology.md`):每个档位一次 chat 请求同时给出两个相位,数字全部取自响应的 `timings` 块。

- prefill = `prompt_n / prompt_ms`;`prompt_ms` 是 prefill 墙钟区间(tokenize/prepare 单独计,不含在内)。
- decode = `(predicted_n - 1) / predicted_ms`;解码预算 256 token,36 次请求全部以 `length` 打满。
- 横轴 = `usage.prompt_tokens`(真实上下文 token 数)。
- 每个请求带唯一 system/user nonce 以打断前缀复用。

**缓存口径**:`root only` 缓存仍可能命中 Qwen 模板前导的约 49 个 token(与内容无关),
所以 prefill 只统计 `cache_n == 0` 的记录(≥8,193 全部 3 次无缓存;≤4,093 每档 1 次无缓存);
decode 不受缓存影响,按全部 3 次重复统计。带缓存记录的 prefill 速率平均高 1-10%(小档位最明显),
全部记录都保留在 `mtp0.json`。

## 结果

| context tokens | prefill tok/s | sd | reps | decode tok/s | sd | reps | decode ms/token | prefill wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 118 | 859.7 ± 0.0 | 1 | 31.09 ± 0.02 | 3 | 32.17 | 0.14 |
| 268 | 1265.7 ± 0.0 | 1 | 31.11 ± 0.02 | 3 | 32.15 | 0.21 |
| 518 | 1432.0 ± 0.0 | 1 | 31.17 ± 0.02 | 3 | 32.08 | 0.36 |
| 1,018 | 1582.2 ± 0.0 | 1 | 31.09 ± 0.01 | 3 | 32.17 | 0.64 |
| 2,043 | 1632.6 ± 0.0 | 1 | 30.99 ± 0.04 | 3 | 32.27 | 1.25 |
| 4,093 | 1649.3 ± 0.0 | 1 | 30.91 ± 0.04 | 3 | 32.36 | 2.48 |
| 8,193 | 1643.3 ± 2.4 | 3 | 30.71 ± 0.03 | 3 | 32.56 | 4.99 |
| 16,393 | 1595.1 ± 1.0 | 3 | 30.42 ± 0.03 | 3 | 32.88 | 10.28 |
| 32,768 | 1510.1 ± 1.2 | 3 | 29.78 ± 0.01 | 3 | 33.58 | 21.70 |
| 65,543 | 1349.1 ± 0.4 | 3 | 28.64 ± 0.02 | 3 | 34.92 | 48.58 |
| 131,068 | 1117.9 ± 0.2 | 3 | 26.65 ± 0.01 | 3 | 37.52 | 117.24 |
| 196,618 | 951.4 ± 0.2 | 3 | 24.98 ± 0.02 | 3 | 40.03 | 206.65 |

## 观察

- **prefill**:4K-8K 是平台(1649 tok/s);之后随上下文单调下降,200K 为 951 tok/s,
  相对平台下降 42%。33K 以上基本是接近线性的慢降,没有台阶
  (TP-2 chunked prefill 步长固定 1024)。
- **decode**:1K 为 31.09 tok/s,200K 为 24.98 tok/s,相对下降 20%;
  每 token 延迟从 32.17 ms 升到 40.03 ms。KV 读取随上下文线性增长是主因。
- **小档位由固定开销主导**:118 token 只有 860 tok/s,到 4K 才接近平台;decode 在 <2K 区间几乎不变。
- **稳定性**:≥8K 的逐次重复 sd ≤ 0.2%(prefill)、≤ 0.1%(decode);小档位的 sd 来自固定开销,不是热噪声。

## 局限

- 仅 C=1 单请求;并发下的 prefill/decode 是另一条曲线。
- MTP0(无投机);MTP3/DFlash 的 decode 曲线需另跑。
- 未开视觉塔,多模态 prefill 未覆盖。
