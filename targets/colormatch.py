import torch


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
        """

        Match the color of the `target` image with the `source` image
        using the **histogram matching** technique

        Using CuPy (for CUDA) and NumPy (for CPU/Non-CUDA device)

        """
        is_cuda = torch.cuda.is_available()

        target = target.numpy()
        source = source.numpy()
        if is_cuda:
            from cucim.skimage.exposure import match_histograms
            import cupy

            target = cupy.asarray(target)
            source = cupy.asarray(source)
        else:
            from skimage.exposure import match_histograms
        matched = match_histograms(target, source, channel_axis=-1)
        matched = cupy.asnumpy(matched) if is_cuda else matched
        del source
        del target
        return (torch.from_numpy(matched),)
