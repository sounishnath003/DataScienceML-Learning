"""
# _* coding: utf8 *_

filename: inference.py

@author: sounishnath
createdAt: 2023-07-26 11:44:16
"""

import torch
from torchf.main import BaseSequence2SequenceModel, Tokenizer, Seq2SeqDataset, create_sequence_tokens
import lightning.pytorch as pl
import torch.utils.data

if __name__ == "__main__":
    model_ckpt_path = "lightning_logs/version_0/checkpoints/epoch=13-step=504.ckpt"
    
    tokenizer = Tokenizer()
    rawdata = open("data/data.txt").readlines()
    data, targets = create_sequence_tokens(tokenizer, rawdata, pad_length=64, ngrams=3)
    print("total paragraphs {0}".format(len(rawdata)))

    litmodel = BaseSequence2SequenceModel.load_from_checkpoint(
        model_ckpt_path, vocab_size=596 #len(tokenizer.all_vocabs)
    )
    litmodel.freeze()
    print(litmodel)

    text = "Washington is keen to draw India closer so that it can act as a"
    tokens=tokenizer.encode(text)
    print(tokens)

    trainer = pl.Trainer()
    pad_length: int = 64

    for i in range(3):
        if len(tokens) < pad_length:
            pads=[tokenizer.pad_token_id] * (pad_length - len(tokens))
            tokens=pads+tokens

        outs = trainer.predict(
            litmodel,
            dataloaders=torch.utils.data.DataLoader(
                dataset=Seq2SeqDataset(
                    data=[tokens],
                    targets=[0]
                )
            ),
        )
        predicted_next_token = outs[0].argmax(dim=1).detach().cpu().item()
        # print(predicted_next_token)
        tokens = tokens + [predicted_next_token]
        print(10*"========")
        print("word", (i + 1), ' '.join(tokenizer.decode(tokens)))
        print(10*"========")

