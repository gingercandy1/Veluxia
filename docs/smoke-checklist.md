# 冒烟测试清单（第一轮，A档：逐个隔离跑）

范围：除 llama-cpp 文字模型、在线翻译外全部本地模型（10项）。
执行：你本机（我无卡无权重）。流程：起后端 → 测一个 → 记结果 →
调 `/health` 确认存活 → 测下一个前重启后端（防显存污染）。

通用模板（Windows curl）：
`curl -s http://127.0.0.1:8765/<prefix>/generate -H "Content-Type: application/json" -d "{\"model_name\":\"<NAME>\",\"extra\":{<EXTRA>}}"`

## 图片（prefix=image）

| # | model_name | 最小 extra | 备注 |
|---|-----------|-----------|------|
| 1 | Flux.1-schnell | `"content":"a cat","width":512,"height":512` | 基线对照组 |
| 2 | SDXL | 同上 | 新 |
| 3 | SD3.5-Medium | 同上 + `"num_inference_steps":10`（先小步验证链路） | 新 |
| 4 | Z-Image-Turbo | `"content":"a cat","width":512,"height":512`（默认9步/g0） | 新 |
| 5 | Qwen-Image-Lightning | 同上（默认8步；**重点看 LoRA 是否报错**） | 新，版本错配裁决点 |
| 6 | rembg-u2net | POST `/image/remove-background`，`"input_path":"<一张本地图绝对路径>"` | 新 |

图生图（2/3/4 任选其一即可）：同上 + `"reference_image":"<绝对路径>"`。

## 动画（prefix=animation）

| # | model_name | 最小 extra |
|---|-----------|-----------|
| 7 | LTX-Video | `"content":"cat running","reference_image_path":"<图>","num_frames":25` |
| 8 | Wan2.2-TI2V | 同上 |

## 语音（prefix=speech）

| # | model_name | 最小 extra |
|---|-----------|-----------|
| 9 | Qwen3-TTS-1.7b-custom | `"content":"你好","voice":"Vivian"`（`mode`错则换 `-base` 名重试并记录） |
| 10 | Ace-Step1.5 | `"content":"epic battle music","duration":10,"batch_size":1`（batch 先改1，默认4太重） |

## 补测

- FILM/Rife 插帧：`POST /image_frame/interpolate`（已补路由），extra 用 `"image_paths":["<图A>","<图B>"],"times_to_interpolate":1`；权重需手动准备（见报错提示），缺权重即记“缺权重”而非 bug。

## 记录表（每行贴一条）

`模型 | 通/挂 | 耗时 | 现象（首句）| traceback 尾20行 | nvidia-smi（挂时）`

## 已知静态问题（跑之前先看，命中即记，不用深挖）

1. **动画参考图键不一致**：image 路由读 `reference_image`，ltx/wan 的 parse 读 `reference_image_path`——UI 传哪个都有一个收不到。冒烟时两个键各试一次，记哪个通。
2. **TTS model_name 待对**：models.json 有 5 个 Qwen3-TTS 名，Generator 的 names 是否一一对应未知；报“未知的生成器”即记。
3. **Qwen-Lightning 必看项**：LoRA 跨版本（8月底座训 / 2512底座用）load 时形状报错=预期内失败，贴错即结论。

回传：记录表 + 命中条目编号，我按 界面/后端路由/生成器/参数传递/环境权重 五维归因。
