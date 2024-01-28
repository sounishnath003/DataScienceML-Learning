# ========================================
#  @project: dcgan-from-scratch
#  @filename: ~/Developer/dcgan-from-scratch/main.py
# ========================================
#  @author: @github/sounishnath003
#  @generatedID: daeab1c2-20ca-4969-8fba-7d341980cd38
#  @createdAt: 27.01.2024 +05:30
# ========================================

import torch
import argparse
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import datasets
from config.logger import Logger
from torch.utils.data import DataLoader, random_split
from config.configuration import HyperParameter
from torchvision import transforms

from trainer.lightning_model import DeepConvolutionGenerativeAdverserialLitNeuralNetwork

if __name__ == "__main__":
    pl.seed_everything(31)
    Logger.get_logger().info(HyperParameter())

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    opts, pipeline_opts = parser.parse_known_args()
    Logger.get_logger().info(opts)

    custom_transforms = transforms.Compose(
        [
            transforms.Resize(HyperParameter.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    fixed_noise = torch.randn(
        size=(HyperParameter.batch_size, HyperParameter.channel_noise, 1, 1)
    )

    dataset = datasets.MNIST(
        root="training_data/dataset/",
        train=True,
        transform=custom_transforms,
        download=True,
    )
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[0.70, 0.30])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=HyperParameter.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=HyperParameter.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    Logger.get_logger().info(
        dict(
            train_dataset_size=len(train_dataset),
            val_dataset_size=len(val_dataset),
        )
    )

    dcgan_model = DeepConvolutionGenerativeAdverserialLitNeuralNetwork(
        channels_noise=HyperParameter.channel_noise,
        channels_img=HyperParameter.channel_img,
        features_generator=HyperParameter.features_generator,
        features_discriminator=HyperParameter.features_discriminator,
        lr=HyperParameter.learning_rate,
    )
    Logger.get_logger().info(dcgan_model)

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                save_top_k=1, save_weights_only=True, mode="min", monitor="d_loss"
            ),
        ],
        fast_dev_run=False if opts.train else True,
        logger=True,
        max_epochs=HyperParameter.epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    trainer.fit(
        model=dcgan_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    out = trainer.test(dcgan_model, val_dataloader)
    Logger.get_logger().info("testing step = {}".format(out))
