import random
import sys
from torch import Tensor
from PIL import Image, ImageOps, ImageEnhance
import torch
from nodes import VAEEncode
from .utils import tensor_to_pil, pil_to_tensor


class Runtime44ImageOverlay:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "overlay"
    CATEGORY = "image/postprocessing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "overlay": ("IMAGE",),
                "overlay_mask": ("MASK",),
                "align_x": (["start", "center", "end"],),
                "align_y": (["start", "center", "end"],),
                "x": (
                    "INT",
                    {
                        "default": 0,
                        "min": sys.maxsize * -1,
                        "max": sys.maxsize,
                    },
                ),
                "y": (
                    "INT",
                    {
                        "default": 0,
                        "min": sys.maxsize * -1,
                        "max": sys.maxsize,
                    },
                ),
                "scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 100.0}),
            }
        }

    def overlay(
        self,
        image: Tensor,
        overlay: Tensor,
        overlay_mask: Tensor,
        align_x: str,
        align_y: str,
        x: int,
        y: int,
        scale: float,
    ) -> Tensor:
        # Convert to pil
        image_pil = tensor_to_pil(image)
        overlay_pil = tensor_to_pil(overlay)

        # Apply the mask (Invert it)
        overlay_pil.putalpha(tensor_to_pil(1.0 - overlay_mask))

        # Rescale the overlay
        if scale != 1.0:
            w, h = overlay_pil.size
            new_size = (int(w * scale), int(h * scale))
            overlay_pil = overlay_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Calculate the anchor point
        anchor_x = (
            0
            if align_x == "start"
            else (
                ((image_pil.width // 2) - (overlay_pil.width // 2))
                if align_x == "center"
                else image_pil.width - overlay_pil.width
            )
        )
        anchor_y = (
            0
            if align_y == "start"
            else (
                ((image_pil.height // 2) - (overlay_pil.height // 2))
                if align_y == "center"
                else image_pil.height - overlay_pil.height
            )
        )

        # Calculate the position
        pos_x = anchor_x + x
        pos_y = anchor_y + y

        # Paste to the original
        image_pil.paste(overlay_pil, (pos_x, pos_y), overlay_pil)

        # Convert to Tensor and return
        return (pil_to_tensor(image_pil),)


class Runtime44ImageResizer:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "resize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 8},
                ),
            }
        }

    def resize(self, image: Tensor, max_resolution: int = 1024):
        image_pil = tensor_to_pil(image)
        output = ImageOps.contain(
            image_pil, (max_resolution, max_resolution), Image.Resampling.LANCZOS
        )

        return (pil_to_tensor(output),)


class Runtime44ImageToNoise:
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("Latent Noise",)
    FUNCTION = "noise"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "colors": ("INT", {"default": 8, "min": 2, "max": 128, "step": 2}),
                "seed": ("INT", {"default": 44, "min": 0, "max": sys.maxsize}),
            }
        }

    def noise(self, image: Tensor, vae, colors: int = 8, seed: int = 44):
        random.seed(seed)
        image_pil = tensor_to_pil(image).quantize(colors).convert("RGBA")
        pixel_data = list(image_pil.getdata())
        random.shuffle(pixel_data)
        r_image_pil = Image.new("RGBA", image_pil.size)
        r_image_pil.putdata(pixel_data)

        r_image_pil = ImageEnhance.Brightness(r_image_pil).enhance(1.0)

        return VAEEncode().encode(vae, pixels=pil_to_tensor(r_image_pil))


class Runtime44ImageEnhance:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "enhance"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": sys.float_info.max,
                        "step": 0.01,
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": sys.float_info.max,
                        "step": 0.01,
                    },
                ),
                "sharpness": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": sys.float_info.max,
                        "step": 0.01,
                    },
                ),
            }
        }

    def enhance(
        self,
        image: Tensor,
        brightness: float = 1.0,
        contrast: float = 1.0,
        sharpness: float = 1.0,
    ):
        image_pil = tensor_to_pil(image)

        enhancers = (
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Sharpness,
        )
        values = (brightness, contrast, sharpness)

        for index, value in enumerate(values):
            if value != 1.0:
                image_pil = enhancers[index](image_pil).enhance(value)

        return (pil_to_tensor(image_pil),)


class Runtime44FilmGrain:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "noise"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["gaussian", "poisson", "speckle"],),
                "sigmas": (
                    "FLOAT",
                    {"default": 0, "min": 0, "max": sys.float_info.max, "step": 0.01},
                ),
                "amount": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": sys.float_info.max, "step": 0.1},
                ),
                "seed": ("INT", {"default": 44, "min": 0, "max": sys.maxsize}),
            }
        }

    def noise(
        self,
        image: Tensor,
        mode: str,
        sigmas: float = 0,
        amount: float = 1.0,
        seed: int = 44,
    ):
        """

        Add film grain-ish to an image.

        Using CuPy (for CUDA) and NumPy (for CPU/Non-CUDA device)

        """
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            import cupy as np
            from cucim.skimage.util import random_noise
        else:
            import numpy as np
            from skimage.util import random_noise

        random.seed(seed)
        image_pil = tensor_to_pil(image)
        source_im = np.asarray(image_pil) if is_cuda else image_pil

        match mode:
            case "gaussian" | "speckle":
                kwargs = {"mean": sigmas}
            case _:
                kwargs = {}

        kwargs = dict(kwargs, seed=seed) if is_cuda else dict(kwargs, rng=seed)

        noise = random_noise(image=source_im, mode=mode, **kwargs)
        noise = np.asnumpy(noise) if is_cuda else noise
        noise_pil = Image.fromarray(np.uint8(noise * 255))

        im = Image.blend(image_pil, noise_pil, alpha=amount)
        del source_im
        del noise

        if is_cuda:
            mempool = np.get_default_memory_pool()
            p_mempool = np.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            p_mempool.free_all_blocks()

        return (pil_to_tensor(im),)
