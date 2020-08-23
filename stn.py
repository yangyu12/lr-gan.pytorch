import torch
from torch import nn


class STNM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, canvas, image, mask, grid):
        tsfm_image = nn.functional.grid_sample(image, grid, align_corners=False)
        tsfm_mask = nn.functional.grid_sample(mask, grid, align_corners=False)
        return tsfm_mask * tsfm_image + (1. - tsfm_mask) * canvas
