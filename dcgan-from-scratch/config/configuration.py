from dataclasses import dataclass


@dataclass
class HyperParameter:
    batch_size: int = 64
    image_size: int = 64
    channel_img: int = 1
    channel_noise: int = 256
    epochs: int = 10
    learning_rate: float = 2e-4

    features_generator: int = 16
    features_discriminator: int = 16
