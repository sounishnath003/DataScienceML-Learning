# ========================================
#  @project: dcgan-from-scratch
#  @filename: ~/Developer/dcgan-from-scratch/trainer/lightning_model.py
# ========================================
#  @author: @github/sounishnath003
#  @generatedID: a32a911d-7f93-4a27-98ae-59583a2f4b96
#  @createdAt: 27.01.2024 +05:30
# ========================================

import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from .model import Generator, Discriminator


class DeepConvolutionGenerativeAdverserialLitNeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        channels_noise: int,
        channels_img: int,
        features_generator: int,
        features_discriminator: int,
        lr: float,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(DeepConvolutionGenerativeAdverserialLitNeuralNetwork, self).__init__(
            *args, **kwargs
        )

        self.lr = lr
        self.channels_noise = channels_noise
        self.channels_img = channels_img
        self.features_generator = features_generator
        self.features_discriminator = features_discriminator

        self.generator = Generator(
            channels_noise=channels_noise,
            channels_img=channels_img,
            feature_dim=features_generator,
        )
        self.discriminator = Discriminator(
            n_channels=channels_img, feature_dim=features_discriminator
        )

        # ..... custom optimizer setup .....
        self.automatic_optimization = False
        self.optG = torch.optim.Adam(
            params=self.generator.parameters(), lr=self.lr, weight_decay=1e-3
        )
        self.optD = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.lr,
            weight_decay=1e-3,
            betas=(0.5, 0.999),
        )

        self.stepper = 0

    def forward(self, x):
        generator_logits = self.generator.forward(x)
        discriminator_logits = self.discriminator.forward(generator_logits)
        return discriminator_logits

    def train_discriminator(self, data, targets):
        # training function: max log(D(x)) + log(1-D(G(z)))
        self.discriminator.train()

        # apply optimizer
        self.optD.zero_grad()

        preds = self.discriminator.forward(data).to(self.device)
        targets = (torch.ones(targets.size(0))).to(self.device)
        real_loss = nn.BCELoss()(preds.view(-1), torch.ones_like(preds.view(-1)))

        fake_data = torch.randn(size=data.size()).to(self.device)
        fake_labels = torch.concat(
            [
                torch.zeros(targets.size(0)).to(self.device),
                torch.ones(targets.size(0)).to(self.device),
            ]
        )
        fake_preds = self.discriminator.forward(
            torch.concat([fake_data, data]).detach()
        ).to(self.device)
        fake_loss = nn.BCELoss()(fake_preds.view(-1), fake_labels)

        loss = (real_loss + fake_loss) / 2
        loss.backward()
        self.optD.step()

        return loss

    def train_generator(self, data, targets):
        # training function: max log(D(G(x)))
        self.generator.train()
        self.discriminator.train()

        # appply optimizer step
        self.optG.zero_grad()

        fake_data = torch.randn(size=(data.size(0), self.channels_noise, 1, 1)).to(
            self.device
        )
        preds = self.forward(fake_data)
        loss = nn.BCELoss()(preds.view(-1), torch.ones_like(preds.view(-1)))

        loss.backward()
        self.optG.step()
        return loss

    def training_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """
        * firstly trains the discriminator with original / real images
        * adds some noisy parameters into the real images
        * trains the generator with the noisy + real images
        * feeds the generated out from generator .... into discriminator
        * compute the loss and accuracy
        """
        data, targets = batch
        # train discriminator
        d_loss = self.train_discriminator(data, targets)
        # train generator
        g_loss = self.train_generator(data, targets)

        loss = d_loss + g_loss
        self.log_dict(
            dict(train_loss=loss, g_loss=g_loss, d_loss=d_loss), prog_bar=True
        )
        return loss

    def validation_step(
        self, batch, batch_id, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        data, targets = batch
        with torch.no_grad():
            fake_data = torch.randn(size=(data.size(0), self.channels_noise, 1, 1)).to(
                self.device
            )
            fake_labels = torch.ones(targets.size(0)).to(self.device)
            fake_generator_logits = self.generator.forward(fake_data).to(self.device)
            preds = self.discriminator.forward(fake_generator_logits)
            loss = nn.BCELoss()(fake_labels, preds.view(-1))

            self.log_dict(dict(val_loss=loss), prog_bar=True)
            return loss

    def test_step(self, batch, batch_id, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(batch, batch, args, kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [self.optG, self.optD]

    def on_train_epoch_end(self) -> None:
        with torch.no_grad():
            fixed_noise = torch.randn(64, self.channels_noise, 1, 1).to(self.device)
            preds = self.generator.forward(fixed_noise)
            generated_img_grid = torchvision.utils.make_grid(preds[:32], normalize=True)
            SummaryWriter(log_dir="lightning_logs/generated_imgs/").add_image(
                "MNIST Generated Img from Noise",
                generated_img_grid,
                global_step=self.stepper,
            )

            self.stepper += 1
