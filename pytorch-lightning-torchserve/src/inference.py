"""
# _* coding: utf8 *_

filename: inference.py

@author: sounishnath
createdAt: 2023-05-23 22:40:09
"""

import torchh

if __name__ == "__main__":
    model=torchh.model.LitImdbNeuralNet(foundation_model=torchh.model.ImdbNeuralNet(),n_classes=1)
    model.eval()

    dataset=torchh.dataset.ImdbDataset(review=["this movie is super cool"], sentiment=[0])
    
    print(dataset[0])