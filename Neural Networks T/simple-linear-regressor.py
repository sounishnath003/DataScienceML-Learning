
import torch
import torch.utils.data
from sklearn import datasets, model_selection


class Dataset:
    def __init__(self, data, targets, tokensizer=None) -> None:
        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)
        self.tokenizer = tokensizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index, :]
        target = self.targets[index]

        return dict(
            data=torch.tensor(data, dtype=torch.float),
            target=torch.tensor(target, dtype=torch.long)
        )


X, targets = datasets.make_classification(
    n_samples=1000, n_features=20, random_state=42)
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(
    X, targets, test_size=0.20, random_state=42, stratify=targets)

train_dataset = Dataset(data=xtrain, targets=ytrain)
test_dataset = Dataset(data=xtest, targets=ytest)


print(train_dataset[238])
print(test_dataset[8])


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=8, shuffle=True)

for data in train_loader:
    print(data["data"])
    print(data["target"])
    break


model = lambda x, w, b: torch.matmul(x, w) + b


W = torch.randn(xtrain.shape[1], requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
learning_rate: float = 1e-2

EPOCHS=100
for epoch in range(EPOCHS+1):
    epoch_loss = 0
    for data in train_loader:
        xtrain = data["data"]
        ytrain = data["target"]

        if W.grad is not None:
            W.grad_zero()

        yhat = model(xtrain, W, b)
        loss = torch.mean(ytrain.view(-1) - yhat.view(-1)).pow(2)
        epoch_loss += loss.item()

        # Computes the gradient of current tensor w.r.t. graph leaves.
        loss.backward()
        with torch.no_grad():
            W = W - learning_rate * W.grad
            b = b - learning_rate * b.grad

        W.requires_grad_()  # update in place
        b.requires_grad_() # update in place

    print (f"epoch {epoch} loss: {epoch_loss / (epoch + 1)}")
