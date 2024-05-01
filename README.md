# Runtime44 ComfyUI Nodes

<p align="center">
    <a href="https://runtime44.com">Runtime44</a> - <a href="https://github.com/runtime44/comfyui_r44_nodes/blob/main/CHANGELOG.md">Changelog</a> - <a href="https://github.com/runtime44/comfyui_r44_nodes/issues">Bug reports</a>
</p>

## Nodes
| Node | Description |
| --- | --- |
| Upscaler | Upscale an image in pixel space using an upscale model. You can select the upscale factor as well as the tile size |
| Color Match | Match the color of an image to the color of a reference |
| Dynamic KSampler | Use multiple samplers during the diffusion process **This node is still a work in progress** |
| Image Overlay | Add an image on top of another, useful for watermarking |
| Image Resizer | Resize an image to a specific resolution, whilst keeping the aspect ratio (usually better when downscaling) |
| Image to Noise | Convert an image into latent noise |
| Mask Sampler | Target the diffusion to a masked region (*Inspired by Impact's SEGS detailer*) |
| Tiled Mask Sampler | Similar to the Mask Sampler, but with a latent tiling system |

## Installation

1. Go to your ComfyUI installation root folder and enter the `custom_nodes` directory
```sh
cd custom_nodes
```

2. Clone this repository
```sh
git clone https://github.com/runtime44/comfyui_r44_nodes.git
```

3. Enter the freshly cloned repository
```sh
cd comfyui_r44_nodes
```

4. Install the python dependencies
```sh
# Using PIP
python -m pip install -r requirements.txt

# Using UV
uv pip install -r requirements.txt
```

5. Restart ComfyUI

## License
These nodes are distributed under the [GNU AGPLv3](./LICENSE.md) license

## Credits
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
