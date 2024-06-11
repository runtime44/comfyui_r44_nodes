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

        if is_cuda:
            from cucim.skimage.exposure import match_histograms
            import cupy

            target = cupy.asarray(target)
            source = cupy.asarray(source)
        else:
            from skimage.exposure import match_histograms

            target = target.numpy()
            source = source.numpy()

        matched = match_histograms(target, source, channel_axis=-1)
        del source
        del target

        im = cupy.asnumpy(matched) if is_cuda else matched
        del matched

        if is_cuda:
            mempool = cupy.get_default_memory_pool()
            p_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            p_mempool.free_all_blocks()

        return (torch.from_numpy(im),)
