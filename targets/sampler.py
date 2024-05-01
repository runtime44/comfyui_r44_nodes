import comfy.samplers
import torch
import math
from nodes import common_ksampler
from .utils import (
    blur_tensor,
    dilate_tensor,
    crop_tensor,
    composite_tensor,
    crop_mask,
    crop_cond,
)


class Runtime44DynamicKSampler:
    """
    Use two different samplers during the same
    diffusion process
    """

    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "first_sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "first_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        first_sampler,
        first_scheduler,
        switch_at_step: int,
        second_sampler,
        second_scheduler,
        denoise,
    ):
        first_k_steps = switch_at_step if switch_at_step < steps else steps
        second_k_steps = steps - switch_at_step
        (latent,) = common_ksampler(
            model=model,
            seed=seed,
            steps=first_k_steps,
            cfg=cfg,
            sampler_name=first_sampler,
            scheduler=first_scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=denoise,
        )
        if switch_at_step > steps:
            return (latent,)
        return common_ksampler(
            model=model,
            seed=seed,
            steps=second_k_steps,
            cfg=cfg,
            sampler_name=second_sampler,
            scheduler=second_scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=denoise,
            disable_noise=True,
        )


class Runtime44MaskSampler:
    """
    Use Mask during the sampling process
    """

    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_feather": ("INT", {"default": 13, "min": 0, "max": 10000}),
                "mask_dilation": ("INT", {"default": 0, "min": -10000, "max": 10000}),
            }
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent,
        mask: torch.Tensor,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name,
        scheduler,
        denoise,
        mask_feather: int,
        mask_dilation: int,
    ):

        mask = self.mutate_mask(mask, mask_dilation, mask_feather)

        # Apply Mask to Latent
        s = latent.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        return common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=s,
            denoise=denoise,
        )

    def mutate_mask(
        self, mask: torch.Tensor, dilation: int, feather: int
    ) -> torch.Tensor:
        # Dilate Mask
        mask = dilate_tensor(mask.numpy(), dilation)
        mask = torch.from_numpy(mask)

        # Feather Mask
        mask = (
            mask.squeeze(0)
            if len(mask.shape) == 4
            else mask.unsqueeze(0) if len(mask.shape) == 2 else mask
        )
        mask = torch.unsqueeze(mask, dim=-1)
        mask = blur_tensor(mask, blur_size=feather)
        return torch.squeeze(mask, dim=-1)


class Runtime44TiledMaskSampler:
    """
    Use Mask during a tiled sampling process
    """

    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 8192, "step": 16},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_feather": ("INT", {"default": 13, "min": 0, "max": 10000}),
                "mask_dilation": ("INT", {"default": 0, "min": -10000, "max": 10000}),
            }
        }

    def sample(
        self,
        model,
        positive,
        negative,
        latent,
        mask: torch.Tensor,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name,
        scheduler,
        tile_size: int,
        denoise: float,
        mask_feather: int,
        mask_dilation: int,
    ):
        latent_tile_size = tile_size // 8
        samples: torch.Tensor = latent.copy()["samples"]
        output = latent.copy()

        tensor_height = samples.size(dim=2)
        tensor_width = samples.size(dim=3)

        rows = math.ceil(tensor_height / latent_tile_size)
        cols = math.ceil(tensor_width / latent_tile_size)

        x = y = 0
        for _ in range(rows):
            if y + latent_tile_size > tensor_height:
                y = tensor_height - latent_tile_size

            for _ in range(cols):
                if x + latent_tile_size > tensor_width:
                    x = tensor_width - latent_tile_size

                lt: dict[str, torch.Tensor] = {}
                lt["samples"] = crop_tensor(samples, (x, y, latent_tile_size))
                m = Runtime44MaskSampler().mutate_mask(
                    mask, mask_dilation, mask_feather
                )
                m = crop_mask(
                    m, (x, y, latent_tile_size), (tensor_width, tensor_height)
                )

                # Skip tiles fully covered by mask
                if m.abs().sum().item() == 0:
                    l = {"samples": lt["samples"]}
                else:
                    # Apply Mask to Latent
                    lt["noise_mask"] = m.reshape((-1, 1, m.shape[-2], m.shape[-1]))

                    pos = crop_cond(positive, (x, y, latent_tile_size))
                    neg = crop_cond(negative, (x, y, latent_tile_size))
                    (l,) = common_ksampler(
                        model=model,
                        seed=seed,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        positive=pos,
                        negative=neg,
                        latent=lt,
                        denoise=denoise,
                    )
                output = composite_tensor(output, l["samples"], (x, y))
                x += latent_tile_size
            x = 0
            y += latent_tile_size

        return (output,)
