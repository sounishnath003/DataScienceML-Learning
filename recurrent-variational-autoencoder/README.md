# Variational Recurrent Auto Encoders (VRAE neural-nets)

Quoting the research paper (1412.6581) it is said -

```
In this paper we propose a model that combines the strengths of RNNs and SGVB:
the Variational Recurrent Auto-Encoder (VRAE). Such a model can be used for
efficient, large scale unsupervised learning on time series data, mapping the time
series data to a latent vector representation. The model is generative, such that
data can be generated from samples of the latent space. An important contribution
of this work is that the model can make use of unlabeled data in order to facilitate
supervised training of RNNs by initialising the weights and network state.
```

This means you can generalize and build a better quality `Timeseries Deep Neural Networks` which can easily applied as `Unsupervised` technique, helps to find out `Timeseries clustering, dimentionality reduction problem, latent vectors (Z-hat), anomaly detection.` etc.

![variational-autoencoder-stack](https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-18-at-12.24.19-AM.png)

## Data and Processing:

Currently, the generated samples are using `torch.randn`. as it is testing out the paper and implementation.

The choice of optimizer proved vital to make the VRAE learn a useful representation, especially
adaptive gradients and momentum are important. In our experiments we used Adam, which is an
optimizer inspired by RMSprop with included momentum and a correction factor for the zero bias,
created by Kingma & Ba (2014).

```
class TimeseriesLightningDataModule(pl.LightningDataModule):
    def __init__(self, series) -> None:
        self.series = series
        self.prepare_data()

    def prepare_data(self) -> None:
        train_, valid_ = random_split(
            TimeseriesDataSet(self.series), lengths=[0.7, 0.3]
        )
        self.train = train_
        self.valid = valid_

    def setup(self, stage: str) -> None:
        self.prepare_data()
        self.train_dataset = TimeseriesDataSet(self.train)
        self.valid_dataset = TimeseriesDataSet(self.valid)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=64, shuffle=False, num_workers=2
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=64, shuffle=False, num_workers=2
        )

```

## Configuration Taken:

```
INFO:[RecurrentVariationAutoEncoderModel]:{'TOTAL_SAMPLES': 1000, 'NUM_OF_FEATURES': 1, 'SEQUENCE_LENGTH': 24, 'HIDDEN_SIZE': 7, 'HIDDEN_LAYER_DEPTH': 2, 'LATENT_DIM': 128, 'DROPOUT': 0.2, 'LEARNING_RATE': 0.01, 'BATCH_SIZE': 64, 'EPOCHS': 2}
```

## PyTorch Model Architecture - Lightning

```
RecurrentVariationAutoEncoderTimeseriesClusteringLit(
  (encoder_model): Encoder(
    (encoder_model): LSTM(1, 7, num_layers=2)
    (dropout): Dropout(p=0.2, inplace=False)
  )
  (lambda_model): LambDa(
    (hidden_2_mean): Linear(in_features=7, out_features=128, bias=True)
    (hidden_2_logvar): Linear(in_features=7, out_features=128, bias=True)
  )
  (decoder_model): Decoder(
    (latent_2_hidden): Linear(in_features=128, out_features=7, bias=True)
    (decoder_model): LSTM(1, 7, num_layers=2)
    (hidden_2_output): Linear(in_features=7, out_features=1, bias=True)
  )
)

  | Name          | Type    | Params
------------------------------------------
0 | encoder_model | Encoder | 728
1 | lambda_model  | LambDa  | 2.0 K
2 | decoder_model | Decoder | 1.6 K
------------------------------------------
4.4 K     Trainable params
0         Non-trainable params
4.4 K     Total params
0.018     Total estimated model params size (MB)

```

## Training and Inference

I have taken `Pytorch Lightning` framework to build the model and inference. with Dataloader set in `LightningDataModule`.

```
Epoch 1: 100%|██████████████████████████████████████| 10/10 [01:17<00:00,  7.72s/it, v_num=0, val_loss=1.150, loss=1.640, kl_loss=0.543]`Trainer.fit` stopped: `max_epochs=2` reached.
Epoch 1: 100%|██████████████████████████████████████| 10/10 [01:17<00:00,  7.72s/it, v_num=0, val_loss=1.150, loss=1.640, kl_loss=0.543]


Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.17it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│          kl_loss          │    0.03689521923661232    │
│           loss            │    1.0564305782318115     │
└───────────────────────────┴───────────────────────────┘
[{'loss': 1.0564305782318115, 'kl_loss': 0.03689521923661232}]
```

## Generative Audio Example: (interesting example by research authors - YT)

1. https://www.youtube.com/watch?v=cu1_uJ9qkHA

## References:

1. https://arxiv.org/pdf/1412.6581.pdf
2. https://github.com/tejaslodaya/timeseries-clustering-vae/
