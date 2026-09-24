"""sherpa-onnx 语音识别：VAD 切句后逐句识别，输出全文、带时间戳的分段和 SRT 字幕。

纯 CPU 推理，不占显存，不参与显存租约（ADR 0003）。
按语言分派模型的规则写在 models.json 的 transcription/Auto 条目里（prefer / auto_detect）。
"""
import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from src.backend.core.model_base import BaseTranscriptionGenerator, load_models_config
from src.backend.core.model_utils import get_temp_dir, hf_download_progress, huggingface_token
from src.shared.settings import PROJECT_ROOT

SAMPLE_RATE = 16000
MODELS_DIR = Path(PROJECT_ROOT) / "models" / "transcription"
VAD_REPO = "csukuangfj/vad"
VAD_FILE = "silero_vad.onnx"
# Whisper 一次最多吃 30 秒；VAD 切出来的句子上限留点余量
MAX_SEGMENT_SECONDS = 20

# 每种结构要下载的文件（只要 int8 权重，仓库里的 fp32 版本和示例音频都不下）
_ARCH_FILES = {
    "sense_voice": ["model.int8.onnx", "tokens.txt"],
    "nemo_transducer": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
    "whisper": ["*-encoder.int8.onnx", "*-decoder.int8.onnx", "*-tokens.txt"],
}


def _catalog() -> dict:
    return load_models_config().get("transcription", {})


def resolve_model(model_name: str, language: str) -> str:
    """把请求里的模型名解析成真正执行识别的模型：具体模型原样返回；
    Auto 按 prefer 顺序挑第一个支持该语言的，语言为 auto 时用 auto_detect。"""
    catalog = _catalog()
    entry = catalog.get(model_name)
    if not isinstance(entry, dict):
        raise ValueError(f"未知的识别模型: {model_name}")
    if "prefer" not in entry:
        return model_name
    if not language or language == "auto":
        return entry["auto_detect"]
    for candidate in entry["prefer"]:
        languages = catalog.get(candidate, {}).get("languages", [])
        if language in languages or "*" in languages:
            return candidate
    raise ValueError(f"没有支持语言 {language} 的识别模型")


def _is_cjk(ch: str) -> bool:
    # 含中日韩标点（U+3000–303F）和全角符号（U+FF00–FFEF），否则"，"后面会多出空格
    return ("　" <= ch <= "ヿ" or "㐀" <= ch <= "鿿"
            or "가" <= ch <= "힯" or "＀" <= ch <= "￯")


def join_segments(texts: list[str]) -> str:
    """中日韩文字之间直接相连，其余语言用空格隔开。"""
    result = ""
    for text in (t.strip() for t in texts):
        if not text:
            continue
        if result and not (_is_cjk(result[-1]) and _is_cjk(text[0])):
            result += " "
        result += text
    return result


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    blocks = []
    for index, seg in enumerate((s for s in segments if s["text"].strip()), 1):
        blocks.append(f"{index}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n"
                      f"{seg['text'].strip()}\n")
    return "\n".join(blocks)


def decode_audio(path: str) -> np.ndarray:
    """任意格式（wav/mp3/m4a/flac/ogg…）解码成 16kHz 单声道 float32。
    用 imageio-ffmpeg 自带的 ffmpeg，不依赖系统里装没装 ffmpeg。"""
    import imageio_ffmpeg

    if not Path(path).is_file():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", path,
           "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"无法解码音频: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


class SherpaASRGenerator(BaseTranscriptionGenerator):
    uses_vram = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded_key: Optional[tuple] = None
        self.target: Optional[str] = None
        self.num_threads = max(1, (os.cpu_count() or 2) // 2)

    def parse_params(self, raw: dict):
        self.audio_path = raw.get("audio_path") or ""
        if not self.audio_path:
            raise ValueError("缺少要识别的音频文件")
        self.language = raw.get("language") or "auto"
        self.target = resolve_model(self.model_name, self.language)
        # Auto 换了语言可能换模型；SenseVoice / Whisper 的语言又是建识别器时定死的。
        # 两者任一变了就先卸掉，ensure_model_loaded 会按新配置重建。
        if self.pipe is not None and self.loaded_key != self._load_key():
            self.unload()
        self.write_srt = bool(raw.get("export_srt", True))
        self.save_path = Path(get_temp_dir(raw.get("output_dir", ""))) / f"transcript_{uuid.uuid4()}.srt"

    def _target_entry(self) -> dict:
        return _catalog()[self.target]

    def _load_key(self) -> tuple:
        # Parakeet 不需要指定语言，换语言不必重建
        arch = self._target_entry()["arch"]
        return self.target, None if arch == "nemo_transducer" else self.language

    def _model_dir(self, repo_id: str) -> Path:
        return MODELS_DIR / repo_id.split("/")[-1]

    def _download(self, repo_id: str, patterns: list[str]):
        from huggingface_hub import snapshot_download

        def on_progress(done, total):
            self.report_load_stage("download", done / total if total else 0.0, repo_id)

        self.report_load_stage("download", detail=repo_id)
        with hf_download_progress(on_progress):
            snapshot_download(repo_id=repo_id, local_dir=str(self._model_dir(repo_id)),
                              allow_patterns=patterns, token=huggingface_token)

    def _check_model_file(self):
        if self.target is None:
            raise RuntimeError("先调用 parse_params 再加载识别模型")
        entry = self._target_entry()
        model_dir = self._model_dir(entry["repo_id"])
        patterns = _ARCH_FILES[entry["arch"]]
        if not all(any(model_dir.glob(p)) for p in patterns):
            print(f"⏬ 正在下载 {self.target}...")
            self._download(entry["repo_id"], patterns)
        if not (self._model_dir(VAD_REPO) / VAD_FILE).exists():
            self._download(VAD_REPO, [VAD_FILE])

    def _load_model(self):
        import sherpa_onnx

        entry = self._target_entry()
        model_dir = self._model_dir(entry["repo_id"])

        def file(pattern: str) -> str:
            return str(next(model_dir.glob(pattern)))

        common = dict(num_threads=self.num_threads, provider="cpu")
        arch = entry["arch"]
        print(f"🔧 正在加载 {self.target}...")
        if arch == "sense_voice":
            self.pipe = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=file("model.int8.onnx"), tokens=file("tokens.txt"),
                language=self._sense_voice_language(), use_itn=True, **common)
        elif arch == "nemo_transducer":
            self.pipe = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=file("encoder.int8.onnx"), decoder=file("decoder.int8.onnx"),
                joiner=file("joiner.int8.onnx"), tokens=file("tokens.txt"),
                model_type="nemo_transducer", **common)
        elif arch == "whisper":
            self.pipe = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=file("*-encoder.int8.onnx"), decoder=file("*-decoder.int8.onnx"),
                tokens=file("*-tokens.txt"),
                language="" if self.language == "auto" else self.language, **common)
        else:
            raise ValueError(f"不支持的识别模型结构: {arch}")
        self.loaded_key = self._load_key()
        print(f"✅ {self.target} 加载完成")

    def _sense_voice_language(self) -> str:
        return self.language if self.language in ("zh", "yue", "en", "ja", "ko") else "auto"

    def _split_speech(self, samples: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """用 Silero VAD 按停顿切句，返回 [(起始秒, 音频)]。"""
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self._model_dir(VAD_REPO) / VAD_FILE)
        config.silero_vad.min_silence_duration = 0.3
        config.silero_vad.max_speech_duration = MAX_SEGMENT_SECONDS
        config.sample_rate = SAMPLE_RATE
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=MAX_SEGMENT_SECONDS * 3)

        window = config.silero_vad.window_size
        segments = []

        def drain():
            while not vad.empty():
                segments.append((vad.front.start / SAMPLE_RATE, np.array(vad.front.samples)))
                vad.pop()

        for offset in range(0, len(samples), window):
            self.check_cancelled()
            vad.accept_waveform(samples[offset:offset + window])
            drain()
        vad.flush()
        drain()
        return segments

    def _recognize(self, samples: np.ndarray) -> str:
        stream = self.pipe.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self.pipe.decode_stream(stream)
        return stream.result.text.strip()

    async def transcribe(self) -> dict:
        samples = decode_audio(self.audio_path)
        pieces = self._split_speech(samples)
        # 很短的口述或 VAD 没检测到人声时，整段直接识别一次，别返回空结果
        if not pieces and 0 < len(samples) <= MAX_SEGMENT_SECONDS * SAMPLE_RATE:
            pieces = [(0.0, samples)]

        segments = []
        for start, audio in pieces:
            self.check_cancelled()
            text = self._recognize(audio)
            if text:
                segments.append({"start": start, "end": start + len(audio) / SAMPLE_RATE, "text": text})

        srt_path = None
        if self.write_srt and segments:
            self.save_path.write_text(to_srt(segments), encoding="utf-8")
            srt_path = str(self.save_path)
        return {
            "text": join_segments([s["text"] for s in segments]),
            "language": self.language,
            "model": self.target,
            "segments": segments,
            "srt_path": srt_path,
        }
