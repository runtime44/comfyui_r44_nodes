from .targets import *

NODE_CLASS_MAPPINGS = {
    "Runtime44Upscaler": Runtime44Upscaler,
    "Runtime44ColorMatch": Runtime44ColorMatch,
    "Runtime44DynamicKSampler": Runtime44DynamicKSampler,
    "Runtime44ImageOverlay": Runtime44ImageOverlay,
    "Runtime44ImageResizer": Runtime44ImageResizer,
    "Runtime44ImageToNoise": Runtime44ImageToNoise,
    "Runtime44MaskSampler": Runtime44MaskSampler,
    "Runtime44TiledMaskSampler": Runtime44TiledMaskSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Runtime44Upscaler": "Runtime44 Upscaler",
    "Runtime44ColorMatch": "Runtime44 Color Match",
    "Runtime44DynamicKSampler": "Runtime44 Dynamic KSampler",
    "Runtime44ImageOverlay": "Runtime44 Image Overlay",
    "Runtime44ImageResizer": "Runtime44 Image Resizer",
    "Runtime44ImageToNoise": "Runtime44 Image To Latent Noise",
    "Runtime44MaskSampler": "Runtime44 Mask Sampler",
    "Runtime44TiledMaskSampler": "Runtime44 Tiled Mask Sampler",
}
