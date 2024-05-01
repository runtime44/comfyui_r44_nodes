from comfy import model_management
import comfy.utils
import torch
import sys


class Runtime44Upscaler:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "upscale_by": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": sys.float_info.max,
                        "step": 0.05,
                    },
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 8192, "step": 8},
                ),
            }
        }

    def upscale(self, upscale_model, image, upscale_by: float, tile_size: int):
        if upscale_by == 0:
            return (image,)
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        free_memory = model_management.get_free_memory(device)
        overlap = 32

        with torch.no_grad():
            oom = True
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        in_img.shape[3],
                        in_img.shape[2],
                        tile_x=tile_size,
                        tile_y=tile_size,
                        overlap=overlap,
                    )
                    print(
                        f"[R44]: Upscaling image with {steps} steps ({tile_size}x{tile_size})\n[R44]: Model scale factor: {upscale_model.scale}"
                    )
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(
                        in_img,
                        lambda a: upscale_model(a),
                        tile_x=tile_size,
                        tile_y=tile_size,
                        overlap=overlap,
                        upscale_amount=upscale_model.scale,
                        pbar=pbar,
                    )
                    size_diff = upscale_by / upscale_model.scale
                    if size_diff != 1:
                        s = comfy.utils.common_upscale(
                            s,
                            width=round(s.shape[3] * size_diff),
                            height=round(s.shape[2] * size_diff),
                            upscale_method="lanczos",
                            crop="disabled",
                        )
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile_size //= 2
                    if tile_size < 128:
                        raise e

            upscale_model.cpu()
            s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)

class Runtime44IterativeUpscaleFactor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_by": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": sys.float_info.max, "step": 0.01
                }),
                "max": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": sys.float_info.max, "step": 0.01
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize, "step": 1
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("factor",)
    FUNCTION = "run"
    CATEGORY = "image/upscaling"

    def run(self, upscale_by: float, max: float, index: int):
        formula = min(upscale_by / (max ** index), max)
        return (0.0 if index > 0 and formula <= 1 else formula,)
