import os
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

def _device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class DiffusersImg2ImgGenerator:
    """
    Minimal img2img generator that can load ONE LoRA at a time and scale it.
    You can extend to multiple LoRAs by loading multiple adapters and calling set_adapters().
    """
    def __init__(self, base_model: str, torch_dtype="auto"):
        self.base_model = base_model
        self.device = _device()

        if torch_dtype == "auto":
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )

        # memory optimizations
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self._loaded_lora = None

    def load_lora(self, lora_path: str, adapter_name: str = "dial"):
        if not lora_path:
            self._loaded_lora = None
            return

        # Avoid reloading same LoRA repeatedly
        if self._loaded_lora == lora_path:
            return

        # Clear any prior adapters (safe-ish)
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass

        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self._loaded_lora = lora_path

    @torch.inference_mode()
    def generate(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        lora_scale: float,
        strength: float,
        cfg: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
    ) -> Image.Image:
        # Apply scale (Diffusers uses set_adapters or fuse_lora depending on version)
        # Most recent adapters route:
        try:
            self.pipe.set_adapters(["dial"], adapter_weights=[float(lora_scale)])
        except Exception:
            # fallback older versions
            try:
                self.pipe.fuse_lora(lora_scale=float(lora_scale))
            except Exception:
                pass

        init = init_image.convert("RGB").resize((width, height), Image.LANCZOS)

        g = torch.Generator(device=self.pipe.device if hasattr(self.pipe, "device") else "cpu")
        g = g.manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            strength=float(strength),
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            generator=g,
        ).images[0]

        return out
