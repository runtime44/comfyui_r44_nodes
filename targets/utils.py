import torch
from torch import Tensor
from PIL import Image
import numpy as np
import comfy.model_management
import cv2
import torchvision
from typing import Any


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil_to_tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def blur_tensor(tensor: Tensor, blur_size: int) -> Tensor:
    """
    @author: https://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/modules/impact/utils.py#L355
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if tensor.ndim == 2:
        tensor = tensor[None, ..., None]
    if tensor.ndim == 3:
        tensor = tensor[..., None]

    blur_size = blur_size * 2 + 1
    small = min(tensor.shape[1], tensor.shape[2])
    if small <= blur_size:
        blur_size = int(small / 2)
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size < 3:
            return tensor

    prev_device = tensor.device
    tensor.to(comfy.model_management.get_torch_device())

    # Blur
    tensor = tensor[:, None, ..., 0]
    blurred_tensor = torchvision.transforms.GaussianBlur(blur_size, sigma=44.0)(tensor)
    blurred_tensor = blurred_tensor[:, 0, ..., None]

    blurred_tensor.to(prev_device)
    return blurred_tensor


def dilate_tensor(tensor: Tensor, factor: int):
    tensor = (
        tensor.squeeze(0).squeeze(0)
        if len(tensor.shape) == 4
        else tensor.squeeze(0) if len(tensor.shape) == 3 else tensor
    )
    if factor == 0:
        return tensor

    kernel = np.ones((abs(factor), abs(factor)), np.uint8)

    return cv2.dilate(tensor, kernel, 1) if factor > 0 else cv2.erode(tensor, kernel, 1)


def crop_tensor(tensor: Tensor, region: tuple[int, int, int]) -> Tensor:
    x, y, size = region
    # Batch Channels Height Width
    return tensor[:, :, y : y + size, x : x + size]


def crop_mask(
    mask: Tensor, region: tuple[int, int, int], latent_size: tuple[int, int]
) -> Tensor:
    x, y, size = region
    # Batch Height Width
    masks = []
    for _ in range(mask.shape[0]):
        mk = tensor_to_pil(mask)
        mk = mk.resize(latent_size, Image.Resampling.BICUBIC)
        mk = mk.crop((x, y, x + size, y + size))
        mk = mk.resize((size, size), Image.Resampling.BICUBIC)

        mk = pil_to_tensor(mk)
        mk = mk.squeeze(-1)
        masks.append(mk)
    return torch.cat(masks, dim=0)


def crop_controlnet(cond_dict: dict, region: tuple[int, int, int]):
    if "control" not in cond_dict:
        return
    x, y, size = region
    cn = cond_dict["control"]
    controlnet = cn.copy()
    cond_dict["control"] = controlnet
    crop_cnet = lambda t: t[:, y : y + size, x : x + size, :]
    while cn is not None:
        # BCHW
        hint: Tensor = controlnet.cond_hint_original
        hint: Tensor = crop_cnet(hint.movedim(1, -1)).movedim(-1, 1)
        hint = torch.nn.functional.interpolate(
            hint.float(),
            size=size,
            mode="bicubic",
            align_corners=False,
        )
        controlnet.cond_hint_original = hint
        cn = cn.previous_controlnet
        controlnet.set_previous_controlnet(cn.copy() if cn is not None else None)
        controlnet = controlnet.previous_controlnet


def crop_cond(cond, region: tuple[int, int, int]):
    cropped = []
    for emb, i in cond:
        cond_dict = i.copy()
        n = [emb, cond_dict]
        crop_controlnet(cond_dict, region)
        cropped.append(n)
    return cropped


def composite_tensor(
    samples_to: dict[str, Any],
    samples_from: Tensor,
    position: tuple[int, int],
    feather: int = 0,
):
    samples_out = samples_to.copy()
    s = samples_to["samples"].clone()
    samples_to = samples_to["samples"]
    x, y = position
    if feather == 0:
        s[:, :, y : y + samples_from.shape[2], x : x + samples_from.shape[3]] = (
            samples_from[:, :, : samples_to.shape[2] - y, : samples_to.shape[3] - x]
        )
    else:
        samples_from = samples_from[
            :, :, : samples_to.shape[2] - y, : samples_to.shape[3] - x
        ]
        mask = torch.ones_like(samples_from)
        for t in range(feather):
            if y != 0:
                mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)

            if y + samples_from.shape[2] < samples_to.shape[2]:
                mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (
                    1.0 / feather
                ) * (t + 1)
            if x != 0:
                mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
            if x + samples_from.shape[3] < samples_to.shape[3]:
                mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (
                    1.0 / feather
                ) * (t + 1)
        rev_mask = torch.ones_like(mask) - mask
        s[:, :, y : y + samples_from.shape[2], x : x + samples_from.shape[3]] = (
            samples_from[:, :, : samples_to.shape[2] - y, : samples_to.shape[3] - x]
            * mask
            + s[:, :, y : y + samples_from.shape[2], x : x + samples_from.shape[3]]
            * rev_mask
        )

    samples_out["samples"] = s
    return samples_out
