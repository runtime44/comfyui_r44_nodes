# Changelog

## v1.1.0 - May 1st, 2024
### Features
- Added new `Iterative Upscale Factor` node to compute an upscale factor for a given index in the upscaling chain

### Changes
- Improved sampling speed when using large mask with `Runtime44TiledMaskSampler`. Areas that are fully covered by the mask are now skipped

### Fixes
- Removed debug `print()` usages

## v1.0.0 - May 1st, 2024
Initial Release
