
import torch
import torch.nn as nn


class NGramTokenBuilder:
    def __init__(self, word_tokens=None, n_gram=2) -> None:
        self.word_tokens = word_tokens
        self.n_gram = n_gram

    def build_ngrams(self):
        ngrams = list()  # [ tuple([ngrams words], target_word_next) ]
        for index in range(self.n_gram, len(self.word_tokens)):
            prev_words = list()
            for prev_index in range(index - self.n_gram, index):
                prev_words.append(self.word_tokens[prev_index])
            ngrams.append((prev_words, self.word_tokens[index]))

        self.__ngrams = ngrams
        return ngrams

    def get_ngrams(self):
        return self.__ngrams


class NGramsLanguageModel(nn.Module):
    def __init__(self, word_tokens=None, vocab_size=None, embedding_dim=None, n_gram=2) -> None:
        super(NGramsLanguageModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(embedding_dim * n_gram, 128)
        self.relu = nn.ReLU()
        self.dense_2 = nn.Linear(128, vocab_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, X):
        x = self.embedding(X)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dense_2(x)
        logits = self.log_softmax(x)
        return logits


class NGramModelTrainer:
    def __init__(self, model: NGramsLanguageModel, word_to_index: dict, epochs=10) -> None:
        self.losses = []
        self.loss_function = nn.NLLLoss()
        self.model = NGramsLanguageModel(
            word_tokens=[], vocab_size=2000, embedding_dim=128, n_gram=2)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        self.epochs = epochs
        self.ngrams = []
        self.word_to_index = word_to_index

    def train_loop(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for context, target in self.ngrams:
                context_idxs = torch.tensor([self.word_to_index[w]
                                            for w in context], dtype=torch.long)
                self.model.zero_grad()
                log_probs = self.model(context_idxs)
                loss = self.loss_function(log_probs, torch.tensor(
                    [self.word_to_index[target]], dtype=torch.long))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.losses.append(total_loss)

            if epoch == 1 or (epoch + 1 == self.epochs) or epoch % 2 == 0:
                print(
                    f"[TRAIN] epoch: {epoch+1}/{self.epochs}, loss: {total_loss:0.4f} ...")

    def losses(self):
        return self.losses


"""
=======================
Output Looks like:
=======================
NGRAMS: [(['upside', 'restart'], 'superb'), (['restart', 'superb'], 'challenging'), (['superb', 'challenging'], 'weve'), (['challenging', 'weve'], 'wiia'), (['weve', 'wiia'], '<UNK>')]
[TRAIN] epoch: 1/20, loss: 0.0000 ...
[TRAIN] epoch: 2/20, loss: 0.0000 ...
[TRAIN] epoch: 3/20, loss: 0.0000 ...
[TRAIN] epoch: 5/20, loss: 0.0000 ...
[TRAIN] epoch: 7/20, loss: 0.0000 ...
[TRAIN] epoch: 9/20, loss: 0.0000 ...
[TRAIN] epoch: 11/20, loss: 0.0000 ...
[TRAIN] epoch: 13/20, loss: 0.0000 ...
[TRAIN] epoch: 15/20, loss: 0.0000 ...
[TRAIN] epoch: 17/20, loss: 0.0000 ...
[TRAIN] epoch: 19/20, loss: 0.0000 ...
[TRAIN] epoch: 20/20, loss: 0.0000 ...
tensor([ 0.5905, -0.1321, -0.1593, -1.3326, -0.5783,  0.1001,  0.9224, -0.9902,
         1.4433,  0.2891, -1.1314, -0.5512,  0.2014, -0.2375,  0.3292, -0.8644,
        -0.8197,  3.2977,  1.7793, -1.8875,  2.3565,  0.2304,  1.5227,  0.9062,
        -0.1407,  0.0893,  0.3166,  0.1282, -0.1472, -0.2574,  1.4378, -0.5091,
         0.1364, -0.6434,  0.8976,  0.7359,  0.5946,  2.3297, -0.7136, -0.1392,
         0.9841,  0.7668,  1.0299,  0.5012,  0.6569, -0.4905, -0.9560, -0.7230,
        -0.4675,  1.1866,  0.0338,  1.4950, -1.0649, -0.2979, -0.6174, -1.4948,
         0.3932,  0.0601,  1.3214, -2.3238, -0.0069, -0.4865,  0.8789, -0.5217],
       grad_fn=<SelectBackward0>)

"""