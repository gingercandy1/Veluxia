<p align="center"><img src="resource/icons/veluxia.png" width="120" alt="Veluxia logo"></p>

# Veluxia

**A local, offline-first AI studio for game assets.** Veluxia brings text, image, animation, voice, music, sound effects, frame interpolation, speech-to-text and translation into one desktop app. Every model runs on your own machine, and the whole thing is tuned to fit a single **8 GB GPU**.

It's built for indie developers and small teams who want to go from an idea to usable game material without juggling a dozen tools, cloud subscriptions or per-image fees: character concepts, idle and run animations, NPC voice lines, background music, UI sound effects, sprite sheets.

---

## Features

### Text and dialogue
- Chat with local LLMs (GGUF via `llama-cpp-python`) for story outlines, character bios, quest text and dialogue trees.
- More than 30 ready-to-download models, grouped by size so you can pick one that fits your hardware: **Qwen 2.5 / 3 / 3.5, Llama 3.x, DeepSeek-R1 distills, Mistral, Phi-4, Gemma 3, Yi, SmolLM2**. They range from a 135M model for weak machines to a 70B flagship.
- Streaming responses, a collapsible "thinking" view for reasoning models, and per-session conversation memory (local vector store).

### Image
| Model | Use |
| --- | --- |
| FLUX.2 Klein 4B (NVFP4) | Fast, high-quality text-to-image |
| SDXL | Versatile base model, seamless tiling for textures |
| SD 3.5 Medium | Strong prompt adherence |
| Z-Image-Turbo (GGUF) | Few-step generation that fits in 8 GB |
| Qwen-Image Lightning (FP8) | Fast generation, strong with Chinese prompts |
| rembg U²-Net / BiRefNet | One-click background removal for sprites and props |
| Real-ESRGAN x4plus / anime 6B | 4× upscaling for photos and illustrations |

### Animation (image-to-video)
- **LTX-Video, LTX-2.3, LTX-2.5**: image-to-video. The LTX-2 models also generate a synchronized audio track, and LTX-2.5 supports distilled fast sampling and automatic duration.
- **Wan 2.2 TI2V 5B**: text- or image-to-video.
- Turn a character concept into idle, walk, attack or other motion clips.

### Frame interpolation and sprite export
- **FILM / RIFE** frame interpolation to smooth choppy animations or raise the frame rate.
- **Sprite sheet export**: pack a frame sequence into one sheet, plus an atlas JSON (frame rects, grid, fps) and numbered PNGs, ready to import into Unity or another engine. It runs on the CPU and never touches the GPU.

### Voice, music and sound effects
- **Qwen3-TTS** (0.6B / 1.7B) in three modes:
  - *Custom voice*: pick from built-in preset speakers.
  - *Voice design*: describe a voice in plain words ("a passionate, energetic 20-year-old girl").
  - *Voice clone*: clone a voice from a short reference clip and its transcript.
- **ACE-Step 1.5**: full-length music generation for background tracks.
- **Stable Audio Open 1.0**: sound effects and short audio clips.

### Speech-to-text
- Local transcription through **sherpa-onnx**. *Auto* mode picks the best engine for the language:
  - **SenseVoice**: Chinese, Cantonese, English, Japanese, Korean.
  - **Parakeet TDT 0.6B v3**: 25 European languages.
  - **Whisper large-v3-turbo**: every other language.

### Workflow helpers
- **Prompt optimizer**: a small local LLM rewrites short ideas into detailed image and video prompts, using a built-in style vocabulary.
- **Prompt translation**: write in your own language and the prompt is translated before it reaches the model (Google Translate with a MyMemory fallback; you can switch it off).
- **Per-model parameter panels**: every model has tailored controls, and your last settings are remembered for each model.
- **Chat-style history**: every generation is kept in a session you can browse, reuse, multi-select and delete.
- **Multilingual UI**: English, 简体中文, 日本語, 한국어, Русский, Español.

---

## Why Veluxia

- **Private and free to run.** No accounts, no API keys, no per-image cost. Your prompts, reference voices and assets never leave your machine.
- **Built for 8 GB GPUs.** The backend keeps only **one large model in VRAM at a time** and swaps automatically when you switch. The models run in fp16 / bf16, FP8, NVFP4 or GGUF, and use CPU offload plus VAE slicing and tiling. Consumer cards like an RTX 3060, 4060 or 4070 can run the full lineup.
- **Safe under load.** A lease system never swaps or unloads a model while a job is running. A second request gets a clear "busy" message instead of corrupting the first job.
- **One app for the whole asset pipeline.** Go from concept (text) to art (image) to motion (animation, interpolation, sprite sheet) to sound (voice, music, SFX) without changing tools.
- **Commercially friendly model choices.** The image lineup was chosen to leave out models whose licenses forbid commercial use. Always check each model's license for your own use case.
- **Responsive while working.** Long jobs run as background tasks with progress, download and load stages, and a working cancel button. The UI never freezes.
- **Easy to extend.** Models are declared in a single `models.json`. You can add a model that uses an existing generator without touching code, and a new generator class plugs in with its own parameter panel.

---

## System requirements

Veluxia is currently **Windows-only** and needs an **NVIDIA GPU** (PyTorch is built for CUDA 12.8).

| | Minimum | Recommended |
| --- | --- | --- |
| OS | Windows 10 64-bit | Windows 11 64-bit |
| GPU | NVIDIA, **8 GB VRAM**, RTX 20-series or newer | NVIDIA RTX 40/50-series, 12 GB+ VRAM |
| GPU driver | Recent driver with CUDA 12.8 support | Latest Game Ready / Studio driver |
| CPU | 6-core, x86-64 | 8+ cores |
| System RAM | 16 GB | **32 GB** (CPU offload keeps parts of the model in RAM) |
| Storage | 60 GB free on an SSD (app plus a few models) | 200 GB+ on an NVMe SSD (for the full lineup) |
| Network | Needed on first use of each model (weights download from Hugging Face) | Fast connection; many weights are several GB |

Notes:
- 8 GB of VRAM is the design target: every model in the default lineup can run on it. More VRAM mainly shortens load times and allows larger resolutions or longer videos.
- Text models tagged **Large** (14B–70B) need 12–48 GB of VRAM and are listed for stronger machines. Stick to *Tiny*, *Small* and *Medium* models on 8 GB.
- Speech-to-text, background removal and sprite export work well on the CPU.

---

## Getting started

Requirements: **Python 3.12+** and [**uv**](https://docs.astral.sh/uv/).

```bat
:: 1. Install dependencies (creates .venv)
uv sync

:: 2. ACE-Step (music) has its own pinned dependencies, so install them separately
uv pip install -r src/backend/core/speech/ACE_Step/requirements.txt

:: 3. Launch the backend and the desktop app together
script\start.bat
```

On first launch the app starts its local backend (`127.0.0.1:8765`). Each model's weights download to `models/` the first time you use it.

Manual start, from the project root:

```bat
.venv\Scripts\python.exe -m src.backend.server --port 8765
.venv\Scripts\python.exe -m src.main
```

Run the tests with `pytest`. Packaging (a separate frontend exe and backend bundle) is done with `python script/package.py front|backend|all`. Pushing a `v*` tag builds both and attaches them to a GitHub Release.

### Release builds and remote backend

The release ships two files: `veluxia-app.exe` and `veluxia-backend.zip`. The zip contains Python and every dependency except PyTorch, which is downloaded (about 3 GB) on first install.

- **Local backend:** in the app, open **Settings → Backend** and choose **Download and install**, or install from a zip you already downloaded.
- **Remote backend:** unzip `veluxia-backend.zip` on the GPU machine and start it with a token:

  ```bat
  set VELUXIA_TOKEN=choose-a-long-random-token
  run_backend.bat --host 0.0.0.0
  ```

  Then in the app choose **Remote backend** and enter `http://<gpu-machine>:8765` and the same token. The backend refuses to listen on a non-local address without a token. The token is sent over plain HTTP, so use this on a trusted LAN or behind an HTTPS reverse proxy.

Remote mode does not yet upload attachments, so features that take an input file (image-to-image, frame interpolation, transcription) only work with a local backend.

---

## Architecture at a glance

```
PySide6 desktop app  ──HTTP──▶  FastAPI backend  ──▶  Generator (one resident model)
 (input bar, param panels,        (routers per modality,      (diffusers / transformers /
  chat history, settings)          job queue, SSE streaming)    llama.cpp / sherpa-onnx)
```

- `src/app`: desktop UI. Network calls run on worker threads.
- `src/backend`: API routers, job manager, model generators.
- `src/shared`: request and response schemas and settings shared by both sides.
- `models.json`: the single source of truth for the model lineup.
- `docs/adr`: architecture decision records.

---

## License

Veluxia's own code is released under the [MIT License](LICENSE).

Each AI model is distributed under **its own license** (Apache-2.0, the Llama Community License, the Stability AI Community License and others). Some models limit commercial use, for example by revenue threshold. Check the license of every model you use before shipping assets commercially.
