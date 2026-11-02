from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MIN_SIDE = 128
MAX_SIDE = 512
MAX_SOURCE_PIXELS = 50_000_000
SUPPORTED_FORMATS = ("PNG", "JPEG", "TIFF")
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".jfif", ".tif", ".tiff")
EIGHT_BIT_MODES = ("L", "LA", "P", "PA", "RGB", "RGBA", "RGBX")
EXIF_ORIENTATION_TAG = 0x0112


class PatchInputError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedImage:
    pixels: np.ndarray
    source_format: str
    source_mode: str
    orientation_corrected: bool

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def needs_crop(self) -> bool:
        return self.width > MAX_SIDE or self.height > MAX_SIDE


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @classmethod
    def from_corners(cls, first: tuple[int, int], second: tuple[int, int]) -> "Crop":
        left, right = sorted((first[0], second[0]))
        top, bottom = sorted((first[1], second[1]))
        return cls(left, top, right - left + 1, bottom - top + 1)

    @classmethod
    def centred(cls, image_width: int, image_height: int) -> "Crop":
        width, height = min(MAX_SIDE, image_width), min(MAX_SIDE, image_height)
        return cls((image_width - width) // 2, (image_height - height) // 2, width, height)


def check_source_size(width: int, height: int) -> None:
    if width < MIN_SIDE or height < MIN_SIDE:
        raise PatchInputError(
            f"The image is {width} × {height} px. Images must be at least {MIN_SIDE} px per side "
            "(images are never enlarged)."
        )
    if width * height > MAX_SOURCE_PIXELS:
        raise PatchInputError(
            f"The image is {width} × {height} px ({width * height / 1e6:.1f} megapixels). The largest supported "
            f"image is {MAX_SOURCE_PIXELS / 1e6:.0f} megapixels; cut out a smaller region first."
        )


def crop_problem(crop: Crop, image_width: int, image_height: int) -> str | None:
    issues = []
    for name, size in (("width", crop.width), ("height", crop.height)):
        if size < MIN_SIDE:
            issues.append(f"the {name} is too small ({size} < {MIN_SIDE})")
        elif size > MAX_SIDE:
            issues.append(f"the {name} is too large ({size} > {MAX_SIDE})")
    if issues:
        return (
            f"The selected crop is {crop.width} × {crop.height} px, but crops must be {MIN_SIDE}–{MAX_SIDE} px "
            f"per side: {' and '.join(issues)}."
        )
    if crop.left < 0 or crop.top < 0 or crop.right > image_width or crop.bottom > image_height:
        return (
            f"The crop at (x={crop.left}, y={crop.top}) extends past the image edge. The image is "
            f"{image_width} × {image_height} px, so x must be 0–{image_width - crop.width} and "
            f"y must be 0–{image_height - crop.height} for a {crop.width} × {crop.height} px crop."
        )
    return None


def crop_pixels(image: LoadedImage, crop: Crop) -> np.ndarray:
    return np.ascontiguousarray(image.pixels[crop.top : crop.bottom, crop.left : crop.right])


def to_rgb(image: Image.Image) -> np.ndarray:
    if image.mode in ("LA", "PA", "RGBA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_image(path: str | Path) -> LoadedImage:
    try:
        with Image.open(path) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise PatchInputError(
                    f"Unsupported file content ({image.format or 'unknown format'}). "
                    "Supported: PNG, JPEG (.jpg, .jpeg, .jfif) and single-page TIFF."
                )
            if getattr(image, "n_frames", 1) != 1:
                kind = "multi-page TIFF" if image.format == "TIFF" else f"multi-frame {image.format}"
                raise PatchInputError(f"Unsupported {kind} ({image.n_frames} frames). Upload a single-page image.")
            if image.mode not in EIGHT_BIT_MODES:
                raise PatchInputError(
                    f"Unsupported {image.format} pixel format '{image.mode}'. Only 8-bit RGB, RGBA, greyscale "
                    "or palette images are supported (not 16-bit, floating-point, bilevel or CMYK)."
                )
            check_source_size(*image.size)
            image.load()
            orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
            oriented = ImageOps.exif_transpose(image)
            return LoadedImage(
                pixels=to_rgb(oriented),
                source_format=image.format,
                source_mode=image.mode,
                orientation_corrected=orientation not in (None, 1),
            )
    except PatchInputError:
        raise
    except UnidentifiedImageError as error:
        raise PatchInputError("The file could not be read as an image. It may be corrupt or not an image at all.") from error
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise PatchInputError(f"The file could not be decoded; it may be corrupt or use an unsupported encoding ({error}).") from error
