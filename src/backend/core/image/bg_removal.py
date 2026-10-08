import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import get_temp_dir


def defringe(out: Image.Image, erode: int = 1, low: float = 0.15, high: float = 0.85) -> Image.Image:
    """收边：去掉遮罩比真实轮廓略宽时留下的一圈半透明背景色（白底图上就是白边）。

    decontaminate 只修颜色、不改 alpha，边缘 alpha 0.1~0.6 的背景像素仍会显出来，所以另外：
    腐蚀 alpha → 两端拉伸（低于 low 归零、高于 high 拉满）→ 边缘 RGB 用内部不透明像素外推。
    """
    import numpy as np
    from scipy import ndimage

    data = np.asarray(out.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, alpha = data[..., :3], data[..., 3]
    if erode > 0:
        alpha = ndimage.grey_erosion(alpha, footprint=np.ones((2 * erode + 1,) * 2))
    alpha = np.clip((alpha - low) / (high - low), 0.0, 1.0)

    solid = ndimage.binary_erosion(alpha > 0.98, iterations=1)
    if solid.any():
        # 每个非实心像素取最近的实心像素颜色
        nearest = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        rgb = np.where(solid[..., None], rgb, rgb[nearest[0], nearest[1]])
    result = np.dstack([rgb, alpha])
    return Image.fromarray((result * 255.0 + 0.5).astype(np.uint8), "RGBA")


class BgRemovalGenerator(BaseImageGenerator):
    """去背景：rembg + u2net（MIT，可商用），输入图 → 透明底 PNG"""
    # rembg 的会话（模型）名。单例按类区分，所以换模型要另写子类而不是靠 models.json 里的名字。
    session_name = "u2net"

    _model_attrs = ("session",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = None

    def _check_model_file(self):
        # u2net.onnx 由 rembg 首次运行时自动下载到 ~/.u2net，无需预处理
        pass

    def _load_model(self):
        if self.session is not None:
            return
        import onnxruntime as ort
        from rembg import new_session
        print(f"🔧 正在加载 rembg-{self.session_name}（首次会自动下载 onnx 权重）...")
        kwargs = {}
        if "CUDAExecutionProvider" in ort.get_available_providers():
            # onnxruntime-gpu 不自带 cuDNN；torch 在 Windows 上 import 时会载入 torch/lib 下的
            # cudnn64_9.dll 等，之后 onnxruntime 按名字就能找到。不先载入，卷积节点逐帧报错
            _ = self.torch
            # BiRefNet fp32 在 1024² 下要吃掉接近 8GB 显存。默认的内存复用规划和按倍数扩容的显存池
            # 会让第二帧起溢出到共享内存，单帧从 1 秒多退化到 20 秒；按需分配 + 启发式选卷积算法
            # 不再预留大块 workspace，实测稳定在 1.2 秒左右
            sess_opts = ort.SessionOptions()
            sess_opts.enable_mem_pattern = False
            cuda = {"arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC"}
            kwargs = {"sess_opts": sess_opts,
                      "providers": [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"]}
        self.session = new_session(self.session_name, **kwargs)
        print(f"✅ rembg-{self.session_name} 推理后端：{self.session.inner_session.get_providers()[0]}")

    async def generate(self) -> Optional[Path | None]:
        self.ensure_model_loaded()
        from rembg import remove

        try:
            img = Image.open(self.input_path)
            # 默认抠图把边缘半透明像素的颜色和黑色混在一起却按非预乘 alpha 保存，边缘发暗；
            # 叠到别的图层上一圈脏边。decontaminate 反推边缘的真实前景色，透明区也填上外推色，
            # 后续放大、引擎双线性采样都不会再把黑色带进边缘
            out = remove(img, session=self.session, decontaminate=True)
            if self.edge_erode >= 0:
                out = defringe(out, erode=self.edge_erode)
            out.save(self.save_path)
            print(f"✅ 去背景完成: {self.save_path.name}")
            return self.save_path
        except Exception as e:
            print(f"❌ 去背景失败: {e}")
        return None

    async def generate_by_image(self) -> Optional[Path | None]:
        # 去背景语义上就是图→图：reference_image 可作为输入源
        if self.ref_image_path and not self.input_path:
            self.input_path = self.ref_image_path
        return await self.generate()

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = self.get_output_dir(self.output_dir)

        self.input_path = raw.get("input_path", "") or raw.get("reference_image", "")
        self.ref_image_path = raw.get("reference_image", "")
        # 收边腐蚀像素数；-1 关闭收边（像素风等需要保留原始 alpha 的场景）
        self.edge_erode = int(raw.get("edge_erode", 1))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"nobg_{str(uuid.uuid4())}.png"


class BiRefNetGenerator(BgRemovalGenerator):
    """去背景升级版：BiRefNet（MIT，可商用），发丝、半透明边缘比 u2net 干净得多，代价是更慢更大。"""
    session_name = "birefnet-general"
