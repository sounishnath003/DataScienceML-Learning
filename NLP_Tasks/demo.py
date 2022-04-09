
import pandas as pd
import torch
import torch.utils.data
from sklearn import model_selection, preprocessing

from cleaner import TextCleanser
from cnn_model import CNNNet
from engine import Engine
from tokenizer import TextTokenizer
from wordembedding import NGramModelTrainer, NGramTokenBuilder, NGramsLanguageModel

if __name__ == "__main__":
    df = pd.read_csv("IMDB Dataset.csv", nrows=100)
    cleanser = TextCleanser(texts=df['review'].values)
    cleanser.remove_urls()
    cleanser.remove_html_tags()
    cleanser.remove_punctuations()
    cleanser.remove_emoji_or_non_unicodes()
    cleanser.remove_stop_words()
    X = cleanser.perform_lemmatize()

    le = preprocessing.LabelEncoder()
    df['review'] = le.fit_transform(df['sentiment'])

    tokenizer = TextTokenizer(
        data=X,
        num_words=20000,
        sequence_length=400
    )

    tokenizer.build_vocabulary()
    tokenizer.vectorize_sentences_by_vocabulary()

    X = tokenizer.pad_sequences()
    targets = df['review'].to_numpy()

    word_tokens = list(tokenizer.vocabulary.keys())
    ngram_token_builder = NGramTokenBuilder(word_tokens=word_tokens, n_gram=2)
    ngrams = ngram_token_builder.build_ngrams()
    print("NGRAMS:", ngram_token_builder.get_ngrams()[-5:])

    ngram_model = NGramsLanguageModel(word_tokens=word_tokens, vocab_size=len(word_tokens), embedding_dim=64, n_gram=2)
    ngram_model_trainer = NGramModelTrainer(model=ngram_model, word_to_index=tokenizer.vocabulary, epochs=20)
    ngram_model_trainer.train_loop()

    # # To get the embedding of a particular word, e.g. "random_vocabulary_word"
    print(ngram_model.embedding.weight[tokenizer.vocabulary["superb"]])

    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(
        X, targets, test_size=0.20, random_state=2022)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(xtrain),
        torch.tensor(ytrain).long()
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(xtest),
        torch.tensor(ytest).long()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True)

    """
    torch.Size([16, 400])
    torch.Size([16])
    """

    model = CNNNet(num_embedding=20000, embedding_dim=400)
    print (model.parameters)

    epochs: int = 50
    lr: float = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = torch.nn.BCELoss()

    Engine.train_loop(model=model, data_loader=train_loader,
                      criterion=criterion, optimizer=optimizer, epochs=epochs)

    Engine.eval_loop(model=model, data_loader=test_loader,
                      criterion=criterion, optimizer=optimizer, epochs=epochs)
