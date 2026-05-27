import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from core.preloader import preloader
from src.shared.settings import PROJECT_ROOT
from src.backend.core.model_base import BaseImageFrameGenerator
from src.backend.core.model_utils import get_temp_dir


def _load_frame_np(path: str, width: int, height: int) -> np.ndarray:
    """图片文件 → float32 numpy (H, W, 3) [0, 1]"""
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _np_to_tensor(arr: np.ndarray, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
    """numpy (H, W, 3) [0,1] → Tensor (1, 3, H, W)"""
    torch = preloader.get("torch")
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """float32 (H, W, 3) [0,1] → PIL Image"""
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))


def _save_frames(
    frames: List[np.ndarray],
    output_dir: Path
) -> List[Path]:
    paths: List[Path] = []
    total = len(frames)
    for i, frame in enumerate(frames):
        save_path = output_dir / f"interp_frame_{i:04d}.png"
        _np_to_pil(frame).save(save_path)
        paths.append(save_path)
        if (i + 1) % 8 == 0 or i == total - 1:
            print(f"  💾 {i+1}/{total} 帧已保存")
    print(f"✅ 共保存 {len(paths)} 帧至 {output_dir}")
    return paths


def _export_video(frames: List[np.ndarray], output_path: Path, fps: int):
    """导出 MP4，优先 mediapy，fallback OpenCV"""
    try:
        import mediapy
        mediapy.write_video(str(output_path), frames, fps=fps)
        print(f"🎬 视频已导出（mediapy）: {output_path}")
        return
    except ImportError:
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"🎬 视频已导出（OpenCV）: {output_path}")
        return
    except ImportError:
        pass
    print("⚠️ 未安装 mediapy / opencv，跳过视频导出。pip install mediapy")


class _FilmInterpolator:
    """FILM TorchScript 推理封装"""

    def __init__(self, model_path: str, device: str, dtype: "torch.dtype"):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"❌ FILM 模型未找到: {model_path}\n"
                "请从 https://github.com/dajes/frame-interpolation-pytorch/releases 下载 .pt 文件"
            )
        print(f"🔧 加载 FILM 模型: {model_path}  ({device}, {dtype})")
        self.device = device
        self.dtype  = dtype

        self.torch = preloader.get("torch")
        self._model = self.torch.jit.load(model_path, map_location="cpu")
        self._model.eval().to(device=device, dtype=dtype)

        print("✅ FILM 加载完成")

    def _infer(self, f0: np.ndarray, f1: np.ndarray, dt: float = 0.5) -> np.ndarray:
        t0   = _np_to_tensor(f0, self.device, self.dtype)
        t1   = _np_to_tensor(f1, self.device, self.dtype)
        dt_t = t0.new_full((1, 1), dt)
        with self.torch.inference_mode():
            out = self._model(t0, t1, dt_t).clamp(0, 1).float()
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def _recursive(self, f0: np.ndarray, f1: np.ndarray, depth: int):
        if depth == 0:
            return
        mid = self._infer(f0, f1)
        yield from self._recursive(f0, mid, depth - 1)
        yield mid
        yield from self._recursive(mid, f1, depth - 1)

    def interpolate(self, frames: List[np.ndarray], times: int) -> List[np.ndarray]:
        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            result.extend(self._recursive(frames[i], frames[i + 1], times))
        result.append(frames[-1])
        return result


class _RifeInterpolator:
    """
    Practical-RIFE 推理封装。

    大动作幻影抑制原理：
      RIFE 使用多尺度双向光流 + 扭曲融合，运动估计比 FILM 更鲁棒；
      scale 参数可在高分辨率时降低光流计算分辨率，进一步减少伪影。
    """

    def __init__(
        self,
        model_dir: str,
        repo_dir:  str,
        device:    "torch.device",
        scale:     float = 1.0,
    ):
        self.device = device
        self.scale  = scale
        self.torch = preloader.get("torch")

        if not Path(repo_dir).exists():
            raise FileNotFoundError(
                f"❌ RIFE 仓库未找到: {repo_dir}\n"
                f"请运行: git clone https://github.com/hzwer/Practical-RIFE {repo_dir}"
            )
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"❌ RIFE 模型目录未找到: {model_dir}\n"
                "请下载模型权重（推荐 v4.22）并放到该目录\n"
                "下载地址: https://github.com/hzwer/Practical-RIFE"
            )

        # 把 Practical-RIFE 仓库加入 sys.path
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        print(f"🔧 加载 RIFE 模型: {model_dir}  ({device})")
        try:
            from models.rife.train_log.RIFE_HDv3 import Model
        except ImportError:
            try:
                from models.rife.train_log.RIFE_HDv3 import Model  # type: ignore
            except ImportError:
                raise ImportError(
                    "无法导入 RIFE Model，请确认 repo_dir 指向正确的 Practical-RIFE 仓库"
                )

        self._model = Model()
        self._model.load_model(model_dir, -1)
        self._model.eval()
        self._model.device()
        print("✅ RIFE 加载完成")

    def _infer(self, f0: np.ndarray, f1: np.ndarray, timestep: float = 0.5) -> np.ndarray:
        t0 = _np_to_tensor(f0, self.device, self.torch.float32)
        t1 = _np_to_tensor(f1, self.device, self.torch.float32)
        with self.torch.inference_mode():
            mid = self._model.inference(t0, t1, timestep=timestep, scale=self.scale)
        return mid.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

    def _recursive(self, f0: np.ndarray, f1: np.ndarray, depth: int):
        if depth == 0:
            return
        mid = self._infer(f0, f1)
        yield from self._recursive(f0, mid, depth - 1)
        yield mid
        yield from self._recursive(mid, f1, depth - 1)

    def interpolate(self, frames: List[np.ndarray], times: int) -> List[np.ndarray]:
        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            result.extend(self._recursive(frames[i], frames[i + 1], times))
        result.append(frames[-1])
        return result


class FILMInterpolationGenerator(BaseImageFrameGenerator):
    """
    帧插值生成器，FILM / RIFE 双后端，接口完全统一。
        backend="film"  精度高，适合小动作
        backend="rife"  幻影少，适合大动作（推荐角色运动场景）
    """
    names = {
                "FILM":{ "tag":"image_frame"},
                "Rife":{ "tag":"image_frame"},
            }

    _FILM_MODEL_PATH = str(Path(PROJECT_ROOT) / "models" / "image_frame" / "film_net" / "Style" / "film_torchscript.pt")
    _RIFE_MODEL_DIR = str(Path(PROJECT_ROOT) / "models" / "image_frame" / "rife" / "train_log")
    _RIFE_REPO_DIR = str(Path(PROJECT_ROOT) / "models" / "image_frame" / "vendors" / "Practical-RIFE")

    def __init__(
        self,
        backend: str = "film",
        film_model_path: Optional[str] = None,
        rife_model_dir:  Optional[str] = None,
        rife_repo_dir:   Optional[str] = None,
        fp16: bool = True,
        rife_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = backend
        self._film_model_path = film_model_path or self._FILM_MODEL_PATH
        self._fp16            = fp16
        self._rife_model_dir  = rife_model_dir or self._RIFE_MODEL_DIR
        self._rife_repo_dir   = rife_repo_dir  or self._RIFE_REPO_DIR
        self._rife_scale      = rife_scale
        self._interpolator    = None

    def _load_model(self):
        if self._interpolator is not None:
            return
        if self.backend == "film":
            dtype = (
                self.torch.float16
                if self._fp16 and self.device == "cuda"
                else self.torch.float32
            )
            self._interpolator = _FilmInterpolator(
                self._film_model_path, self.device, dtype
            )
        else:
            self._interpolator = _RifeInterpolator(
                self._rife_model_dir,
                self._rife_repo_dir,
                self.device,
                self._rife_scale,
            )

    async def interpolate_images(self) -> List[Path]:
        self.ensure_model_loaded()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_a_path = self.image_paths[0]
        image_b_path = self.image_paths[1]

        fa = _load_frame_np(image_a_path, self.width, self.height)
        fb = _load_frame_np(image_b_path, self.width, self.height)
        total = 2 ** self.times_to_interpolate + 1
        print(f"🎞️ [{self.backend.upper()}] 开始插值 → {total} 帧  ({self.width}×{self.height})")

        all_frames = self._interpolator.interpolate([fa, fb], self.times_to_interpolate)
        frame_paths = _save_frames(all_frames, output_dir)

        if self.export_video:
            frames_u8 = [np.clip(f * 255, 0, 255).astype(np.uint8) for f in all_frames]
            _export_video(frames_u8, output_dir / f"{self.backend}_output.mp4", self.fps)
        return frame_paths

    async def interpolate_sequence(self) -> List[Path]:
        if len(self.image_paths) < 2:
            raise ValueError("至少需要提供 2 张图片")

        self.ensure_model_loaded()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        key_frames = [_load_frame_np(p, self.width, self.height) for p in self.image_paths]
        print(f"✅ 加载 {len(key_frames)} 张关键帧")

        all_frames = self._interpolator.interpolate(key_frames, self.times_to_interpolate)
        print(f"🎞️ [{self.backend.upper()}] 插值完成，共 {len(all_frames)} 帧")

        frame_paths = _save_frames(all_frames, output_dir)

        if self.export_video:
            frames_u8 = [np.clip(f * 255, 0, 255).astype(np.uint8) for f in all_frames]
            _export_video(frames_u8, output_dir / f"{self.backend}_sequence.mp4", self.fps)
        return frame_paths

    def parse_params(self, raw):
        self.image_paths          = raw.get("image_paths", [])
        self.output_dir           = get_temp_dir(raw.get("output_dir", ""))
        self.times_to_interpolate = raw.get("times_to_interpolate", 1)
        self.width                = raw.get("width", 1024)
        self.height               = raw.get("height", 1024)
        self.fps                  = raw.get("fps", 8)
        self.export_video         = raw.get("export_video", True)
        self.model_path           = raw.get("model_path", None)