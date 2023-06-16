import torch
import torch.nn as nn

from src.vision_transformer.modules.vit_input_layer import VitInputLayer

def test_vit_input_layer() -> None:

    in_channels: int = 3
    image_size: int = 32
    num_batch: int = 16

    imgs = torch.randn(num_batch, in_channels, image_size, image_size)
    vit_input_layer = VitInputLayer()
    x = vit_input_layer.forward(imgs)
    assert x.shape == (num_batch, 5, 384)
