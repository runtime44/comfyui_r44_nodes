import torch
import numpy as np
from skimage.exposure import match_histograms


class Runtime44ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "image"

    def match(self, source: torch.Tensor, target: torch.Tensor):
        target = target.numpy()
        source = source.numpy()
        matched = match_histograms(target, source, channel_axis=None)
        return (torch.from_numpy(matched),)
