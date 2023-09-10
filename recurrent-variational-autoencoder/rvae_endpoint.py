"""
# _* coding: utf8 *_

filename: rvae_handler.py

@author: sounishnath
createdAt: 2023-09-10 09:59:06
"""

"""
ModelHandler defines a custom model handler.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from ts.torch_handler.base_handler import BaseHandler

# ###### < lightning model defination > ####### #


class ModelHandler(BaseHandler):
    class TimeseriesDataSet:
        def __init__(self, series) -> None:
            self.series = series

        def __len__(self):
            return len(self.series)

        def __getitem__(self, item):
            return {
                "timeseries_sequence_sample": torch.tensor(
                    self.series[item], dtype=torch.float
                )
            }

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

    ####### Global Constants #########
    # logging.basicConfig(level=logging.NOTSET)
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
            super(ModelHandler.Encoder, self).__init__()

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
            _, (h_end, c_end) = self.encoder_model(tensors)
            h_end = self.dropout(h_end)
            return h_end[-1, :, :], (h_end, c_end)

    class LambDa(nn.Module):
        def __init__(self, hidden_size: int, latent_dim: int, *args, **kwargs) -> None:
            super(ModelHandler.LambDa, self).__init__()

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
            super(ModelHandler.Decoder, self).__init__()

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

            x_hidden = torch.stack([x_hidden for _ in range(self.hidden_layer_depth)])

            decoder_output, _ = self.decoder_model.forward(
                self.decoder_inputs, (x_hidden, states[1])
            )

            decoder_output = self.hidden_2_output.forward(decoder_output)

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
            *args,
            **kwargs
        ) -> None:
            super(
                ModelHandler.RecurrentVariationAutoEncoderTimeseriesClusteringLit, self
            ).__init__()

            self.sequence_length = sequence_length
            self.num_of_features = num_of_features
            self.hidden_size = hidden_size
            self.hidden_layer_depth = hidden_layer_depth
            self.latent_dim = latent_dim
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.dropout = dropout

            self.encoder_model = ModelHandler.Encoder(
                num_of_features=self.num_of_features,
                hidden_size=self.hidden_size,
                hidden_layer_depth=self.hidden_layer_depth,
                latent_dim=self.latent_dim,
                dropout=self.dropout,
            )
            self.lambda_model = ModelHandler.LambDa(
                hidden_size=self.hidden_size, latent_dim=self.latent_dim
            )
            self.decoder_model = ModelHandler.Decoder(
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
            latent_logits, logvar_logits = self.lambda_model.forward(logits)
            X_hat = self.decoder_model.forward(latent_logits, (h_end, c_end))
            return (
                X_hat.reshape(
                    self.batch_size, self.sequence_length, self.num_of_features
                ),
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

        def training_step(self, batch, batch_idx, *args, **kwargs):
            X_tensors = batch["timeseries_sequence_sample"]
            X_hat, latent_logits, logvar_logits = self.forward(batch)
            loss, recon_loss, kl_loss = self._compute_loss(
                X_hat, X_tensors, latent_logits, logvar_logits
            )
            self.log("loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False)
            self.log(
                "kl_loss", kl_loss.item(), prog_bar=True, on_epoch=True, on_step=False
            )

            return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

        def validation_step(self, batch, batch_idx, *args, **kwargs):
            X_tensors = batch["timeseries_sequence_sample"]
            X_hat, latent_logits, logvar_logits = self.forward(batch)
            loss, recon_loss, kl_loss = self._compute_loss(
                X_hat, X_tensors, latent_logits, logvar_logits
            )
            self.log(
                "val_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False
            )

            return {
                "val_loss": loss,
                "val_recon_loss": recon_loss,
                "val_kl_loss": kl_loss,
            }

        def test_step(self, batch, batch_idx, *args, **kwargs):
            return self.training_step(batch, batch_idx)

        def configure_optimizers(self):
            opt = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=1e-5
            )
            sch = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.55, total_iters=5
            )
            return {"optimizer": opt, "lr_scheduler": sch}

    # ###### < lightning model defination > ####### #

    """
    A custom model handler implementation.
    """

    @dataclass
    class Configuration:
        TOTAL_SAMPLES: int = 1000
        NUM_OF_FEATURES: int = 1
        SEQUENCE_LENGTH: int = 24
        HIDDEN_SIZE: int = int(SEQUENCE_LENGTH * 0.30) or 2
        HIDDEN_LAYER_DEPTH: int = 2
        LATENT_DIM: int = 128
        DROPOUT: float = 0.20
        LEARNING_RATE: float = 1e-2
        BATCH_SIZE: int = 64
        EPOCHS: int = 2

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

        self.manifest = context.manifest

        try:
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            self.device = torch.device(
                "cuda:" + str(properties.get("gpu_id"))
                if torch.cuda.is_available()
                else "cpu"
            )

            # Read model serialize/pt file
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self._model = ModelHandler.RecurrentVariationAutoEncoderTimeseriesClusteringLit.load_from_checkpoint(
                model_pt_path,
                sequence_length=ModelHandler.Configuration.SEQUENCE_LENGTH,
                num_of_features=ModelHandler.Configuration.NUM_OF_FEATURES,
                hidden_size=ModelHandler.Configuration.HIDDEN_SIZE,
                hidden_layer_depth=ModelHandler.Configuration.HIDDEN_LAYER_DEPTH,
                latent_dim=ModelHandler.Configuration.LATENT_DIM,
                batch_size=ModelHandler.Configuration.BATCH_SIZE,
                learning_rate=ModelHandler.Configuration.LEARNING_RATE,
                dropout=ModelHandler.Configuration.DROPOUT,
            )
            self._model.eval()
            self.trainer = pl.Trainer()

        except Exception as e:
            print("Error occured while loading lightning model {}".format(e))

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        print(
            "received inputs from http.calls = {0}. After processing {1}".format(
                data, preprocessed_data
            )
        )
        return preprocessed_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        input_np = [
            torch.randn(
                ModelHandler.Configuration.SEQUENCE_LENGTH,
                ModelHandler.Configuration.NUM_OF_FEATURES,
            )
            .detach()
            .cpu()
            .numpy()
            for _ in range(100)
        ]

        predict_dataloader = DataLoader(
            dataset=ModelHandler.TimeseriesDataSet(input_np),
            batch_size=64,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )

        # Do some inference call to engine here and return output
        model_output = self.trainer.test(
            model=self._model, dataloaders=predict_dataloader, verbose=True
        )
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
