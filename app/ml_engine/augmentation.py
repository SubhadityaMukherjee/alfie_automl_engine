"""Default augmentation pipelines for the image and video dataloaders.

The pipelines are built from torchvision v2 transforms and mirror the
illustration snippets used to spec them: random border padding
(``v2.Pad`` with paddings 3/10/50), random scale variation
(``v2.Resize`` with shortest-edge sizes 30/50/100 or the original size),
a perspective warp (``v2.RandomPerspective`` with ``distortion_scale=0.6``)
and random grayscale (``v2.RandomGrayscale``).

They are applied to the **training split only**; validation and test
dataloaders stay deterministic so Optuna's validation-loss selection
remains comparable across trials.
"""

import logging
from typing import Any

import torch
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

# Border paddings (pixels) sampled for the pad augmentation.
PAD_CHOICES: tuple[int, ...] = (3, 10, 50)
# Shortest-edge sizes sampled for the scale-variation augmentation
# (``v2.Identity`` keeps the original size with the same probability).
RESIZE_CHOICES: tuple[int, ...] = (30, 50, 100)
# Probability that a given sample receives a random pad at all.
PAD_PROB: float = 0.5
# RandomPerspective parameters (the v2 docs' illustration values).
PERSPECTIVE_DISTORTION_SCALE: float = 0.6
PERSPECTIVE_PROB: float = 0.5
# Probability a sample is converted to grayscale. ``RandomGrayscale``
# keeps the 3-channel layout, so HF image processors downstream of the
# dataloaders keep working (unlike ``v2.Grayscale`` which drops to 1).
GRAYSCALE_PROB: float = 0.1


def _augmentation_stages() -> list[Any]:
    """Shared v2 stages: pad -> perspective warp -> random resize -> grayscale."""
    return [
        v2.RandomApply(
            [v2.RandomChoice([v2.Pad(padding=p) for p in PAD_CHOICES])],
            p=PAD_PROB,
        ),
        v2.RandomPerspective(
            distortion_scale=PERSPECTIVE_DISTORTION_SCALE, p=PERSPECTIVE_PROB
        ),
        v2.RandomChoice([v2.Resize(size) for size in RESIZE_CHOICES] + [v2.Identity()]),
        v2.RandomGrayscale(p=GRAYSCALE_PROB),
    ]


class RestoreSize:
    """Resize a clip back to its native ``(H, W)`` after random rescaling.

    The video collate function stacks clips with ``torch.stack``, so every
    clip in a batch must share the same size; this trailing step guarantees
    the scale-variation augmentation never changes a clip's final size.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return v2.functional.resize(clip, self.size)


def build_image_augmentation() -> v2.Compose:
    """Default augmentation pipeline for PIL image samples.

    Input and output are 3-channel RGB PIL images, so HF image processors
    that run in the collate functions keep working unchanged.
    """
    return v2.Compose(_augmentation_stages())


def build_video_augmentation(native_size: tuple[int, int]) -> v2.Compose:
    """Default augmentation pipeline for decoded video clips.

    Apply the pipeline to a whole ``(T, C, H, W)`` clip tensor: v2 samples
    one parameter set per call, so the pad/perspective/grayscale choices
    stay temporally consistent across the clip's frames.

    ``native_size`` is the clip's decoded ``(H, W)``; a trailing
    :class:`RestoreSize` puts every clip back there so batches stack.
    """
    return v2.Compose(_augmentation_stages() + [RestoreSize(native_size)])
