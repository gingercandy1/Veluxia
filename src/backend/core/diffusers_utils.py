"""diffusers 生成器共用的加载步骤：GGUF 量化 transformer、LoRA、显存卸载。

只放普通函数，不做基类 / mixin：各生成器的管道结构差异大，按需组合调用即可。
torch / diffusers 都在函数内导入，保持本模块可在无 GPU 环境下被契约测试导入。
"""
from dataclasses import dataclass
from pathlib import Path

from src.backend.core.model_utils import ensure_file
from src.shared.settings import PROJECT_ROOT

OFFLOAD_MODES = ("model", "group", "sequential")
# LoRA 按来源仓库集中存放：同一仓库常给多个底座出权重（如 Lightning），不必各存一份
LORA_ROOT = Path(PROJECT_ROOT) / "models" / "lora"


def gguf_filename(filename: str | None) -> str | None:
    """models.json 的 filename 是 .gguf 才走量化链路，否则返回 None 表示用官方全量权重。"""
    return filename if str(filename or "").endswith(".gguf") else None


def load_gguf_transformer(model_cls, gguf_path: Path, base_dir: Path):
    """从 GGUF 单文件构建量化 transformer，权重在内存和显存里都保持压缩。

    结构配置一律取本地官方骨架的 transformer/config.json：不传 config 时
    from_single_file 会联网到 hub 上猜配置，离线就加载失败。
    """
    import torch
    from diffusers import GGUFQuantizationConfig

    print(f"⚡ 正在以 GGUF 量化加载 transformer（{Path(gguf_path).name}）...")
    return model_cls.from_single_file(
        str(gguf_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=str(base_dir),
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )


@dataclass(frozen=True)
class LoraSpec:
    repo_id: str
    weight_name: str
    name: str
    scale: float = 1.0

    @property
    def repo_dir(self) -> Path:
        return LORA_ROOT / self.repo_id.replace("/", "--")

    @property
    def path(self) -> Path:
        return self.repo_dir / self.weight_name


def parse_loras(model_extra: dict | None) -> list[LoraSpec]:
    """读取 models.json 里的 "loras" 列表；字段缺失直接报错，免得加载到一半才发现配错。"""
    specs = []
    for index, item in enumerate((model_extra or {}).get("loras", [])):
        missing = [key for key in ("repo_id", "weight_name") if not item.get(key)]
        if missing:
            raise ValueError(f"loras[{index}] 缺少字段：{', '.join(missing)}")
        specs.append(LoraSpec(
            repo_id=item["repo_id"],
            weight_name=item["weight_name"],
            # set_adapters 按名字区分多个 LoRA，没写名字时用序号保证不重名
            name=item.get("name") or f"lora_{index}",
            scale=float(item.get("scale", 1.0)),
        ))
    return specs


def ensure_loras(loras: list[LoraSpec]):
    """LoRA 下载到 models/ 下而不是 HF 缓存，和其它权重一起管理，离线也能用。"""
    for lora in loras:
        ensure_file(lora.repo_id, lora.weight_name, lora.repo_dir)


def apply_loras(pipe, loras: list[LoraSpec]):
    """按配置顺序叠加 LoRA 并设置各自权重；必须在 offload 之前调用，否则权重挂不到正确设备。"""
    if not loras:
        return
    for lora in loras:
        pipe.load_lora_weights(str(lora.path.parent), weight_name=lora.path.name,
                               adapter_name=lora.name)
    pipe.set_adapters([lora.name for lora in loras],
                      adapter_weights=[lora.scale for lora in loras])


def apply_offload(pipe, mode: str = "model", device: str = "cuda"):
    """显存卸载策略，同时打开 VAE 分片 / 分块解码（8GB 下高分辨率解码的峰值主要在 VAE）。

    - model：整模块在 CPU / GPU 间搬运，最快；但它靠存储指针互换，和 torchao 量化
      tensor 子类不兼容（报 "storage of a tensor on device cuda:0 ..."）。
    - group：按叶子层搬运参数，量化权重可用，显存最省但更慢。
    - sequential：逐层搬运，最慢，只在前两者都放不下时用。
    """
    if mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "group":
        import torch
        pipe.enable_group_offload(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
        )
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"未知的 offload 模式：{mode}，可选 {OFFLOAD_MODES}")
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        vae.enable_slicing()
        vae.enable_tiling()
