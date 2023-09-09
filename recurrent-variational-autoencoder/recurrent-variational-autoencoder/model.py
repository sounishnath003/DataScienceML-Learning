"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2023-08-15 22:15:21
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader, random_split

from .dataset import TimeseriesDataSet

####### Global Constants #########
# logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("[RecurrentVariationAutoEncoderModel]")
####### Global Constants #########


class Encoder(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        hidden_size: int,
        hidden_layer_depth: int,
        latent_dim: int,
        dropout=0.20,
        *args,
        **kwargs
    ) -> None:
        super(Encoder, self).__init__()

        self.num_of_features = num_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.encoder_model = nn.LSTM(
            self.num_of_features,
            self.hidden_size,
            self.hidden_layer_depth,
            batch_first=False,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, tensors):
        logger.debug("size of tensors = {}".format(tensors.size()))
        _, (h_end, c_end) = self.encoder_model(tensors)
        logger.debug("h_end size = {0}".format(h_end.size()))
        h_end = self.dropout(h_end)
        logger.debug("c_end size = {0}".format(c_end.size()))
        return h_end[-1, :, :], (h_end, c_end)


class LambDa(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, *args, **kwargs) -> None:
        super(LambDa, self).__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.hidden_2_mean = nn.Linear(self.hidden_size, self.latent_dim)
        self.hidden_2_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        nn.init.xavier_uniform_(self.hidden_2_mean.weight)
        nn.init.xavier_uniform_(self.hidden_2_logvar.weight)

    def forward(self, tensors):
        mean_logits = self.hidden_2_mean(tensors)
        logvar_logits = self.hidden_2_logvar(tensors)

        if self.training:
            std = torch.exp(0.50 * logvar_logits)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean_logits), logvar_logits

        return mean_logits, logvar_logits


class Decoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_size: int,
        hidden_layer_depth: int,
        latent_dim: int,
        output_size: int,
        dropout: float = 0.20,
        *args,
        **kwargs
    ) -> None:
        super(Decoder, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.dropout = dropout
        self.batch_size = 64

        self.latent_2_hidden = nn.Linear(self.latent_dim, self.hidden_size)
        self.decoder_model = nn.LSTM(
            1, self.hidden_size, self.hidden_layer_depth, batch_first=False
        )
        self.hidden_2_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(
            self.sequence_length, self.batch_size, 1, requires_grad=True
        ).to("mps")
        self.c_0 = torch.zeros(
            self.hidden_layer_depth,
            self.batch_size,
            self.hidden_size,
            requires_grad=True,
        )

        nn.init.xavier_uniform_(self.latent_2_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_2_output.weight)

    def forward(self, tensors, states=None):
        x_hidden = F.relu(self.latent_2_hidden(tensors))
        logger.debug("size of latent2hidden = {0}".format(x_hidden.size()))

        x_hidden = torch.stack([x_hidden for _ in range(self.hidden_layer_depth)])

        decoder_output, _ = self.decoder_model.forward(
            self.decoder_inputs, (x_hidden, states[1])
        )
        logger.debug("decoder_outputs size = {0}".format(decoder_output.size()))

        decoder_output = self.hidden_2_output.forward(decoder_output)
        logger.debug("decoder_outputs size = {0}".format(decoder_output.size()))

        return decoder_output


class RecurrentVariationAutoEncoderTimeseriesClusteringLit(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        num_of_features: int,
        hidden_size: int,
        hidden_layer_depth: int = 2,
        latent_dim: int = 128,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        dropout: float = 0.20,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(RecurrentVariationAutoEncoderTimeseriesClusteringLit, self).__init__()

        self.sequence_length = sequence_length
        self.num_of_features = num_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.encoder_model = Encoder(
            num_of_features=self.num_of_features,
            hidden_size=self.hidden_size,
            hidden_layer_depth=self.hidden_layer_depth,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )
        self.lambda_model = LambDa(
            hidden_size=self.hidden_size, latent_dim=self.latent_dim
        )
        self.decoder_model = Decoder(
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            hidden_layer_depth=self.hidden_layer_depth,
            latent_dim=self.latent_dim,
            output_size=self.num_of_features,
            dropout=self.dropout,
        )

        # self.save_hyperparameters()

    def _common_step(self, tensors):
        logits, (h_end, c_end) = self.encoder_model.forward(
            tensors.permute(1, 0, 2)
        )  # [ BS x SEQ x FEAT ] --> [ SEQ x BS x FEAT ]
        logging.debug("size of logits = {}".format(logits.size()))
        logging.debug("size of h_end = {}".format(h_end.size()))
        logging.debug("size of c_end = {}".format(c_end.size()))
        latent_logits, logvar_logits = self.lambda_model.forward(logits)
        logging.debug("size of latent_logits = {}".format(latent_logits.size()))
        logging.debug("size of logvar_logits = {}".format(logvar_logits.size()))
        X_hat = self.decoder_model.forward(latent_logits, (h_end, c_end))
        logging.debug("size of X_hat = {}".format(X_hat.size()))
        return (
            X_hat.reshape(self.batch_size, self.sequence_length, self.num_of_features),
            latent_logits,
            logvar_logits,
        )

    def _compute_loss(self, X_hat, X_original, latent_logits, logvar_logits):
        kl_loss = -0.50 * torch.mean(
            1 + logvar_logits - latent_logits.pow(2) - logvar_logits.exp()
        )
        recon_loss = torchmetrics.MeanSquaredError()(
            X_hat.detach().cpu(), X_original.detach().cpu()
        )
        return (
            torch.autograd.Variable(recon_loss + kl_loss, requires_grad=True),
            torch.autograd.Variable(recon_loss, requires_grad=True),
            torch.autograd.Variable(kl_loss, requires_grad=True),
        )

    def forward(self, batch):
        X_tensors = batch["timeseries_sequence_sample"]
        X_hat, latent_logits, logvar_logits = self._common_step(X_tensors)
        return X_hat, latent_logits, logvar_logits

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        X_tensors = batch["timeseries_sequence_sample"]
        X_hat, latent_logits, logvar_logits = self.forward(batch)
        loss, recon_loss, kl_loss = self._compute_loss(
            X_hat, X_tensors, latent_logits, logvar_logits
        )
        self.log("loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False)
        self.log("kl_loss", kl_loss.item(), prog_bar=True, on_epoch=True, on_step=False)

        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def validation_step(
        self, batch, batch_idx, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        X_tensors = batch["timeseries_sequence_sample"]
        X_hat, latent_logits, logvar_logits = self.forward(batch)
        loss, recon_loss, kl_loss = self._compute_loss(
            X_hat, X_tensors, latent_logits, logvar_logits
        )
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False)

        return {"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_loss": kl_loss}

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.55, total_iters=5)
        return {"optimizer": opt, "lr_scheduler": sch}
