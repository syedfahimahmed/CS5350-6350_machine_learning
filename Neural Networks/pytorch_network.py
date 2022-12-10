import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class BankNoteDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)
        self.data.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
        self.X = self.data.iloc[:, :-1].values
        self.Y = self.data.iloc[:, -1].values
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_data = BankNoteDataset('bank-note/train.csv')
test_data = BankNoteDataset('bank-note/test.csv')

device = "cuda" if torch.cuda.is_available() else "cpu"


# Create data loaders.
batch_size = 10
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(torch.reshape(pred, y.shape), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss.append(loss.item())

    epoch_loss = np.mean(train_loss)
    #print(f"Loss: {epoch_loss:>8f}")
    return epoch_loss


def test(model, X, Y):
    model.eval()
    #for batch, (X, Y) in enumerate(dataloader):
    with torch.no_grad():
        X, y = X.to(device), Y.to(device)
        pred = model(X.float())
        pred = (pred > 0.5).float()
        error = torch.mean((pred != y).float())

    #error = np.mean(test_error)
    return error


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
activations = [(nn.ReLU(), init_he, "ReLU"), (nn.Tanh(), init_xavier, "Tanh")]

for ac_fn, init_fn, ac_name in activations:
    print(f"activation function {ac_name}")
    for width in widths:
        for depth in depths:

            print(f"{depth}-depth, {width}-width network:\n-------------------------------")


            # Define model
            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.input = nn.Sequential(nn.Linear(4, width), ac_fn)
                    self.body = nn.ModuleList([])
                    for i in range(depth - 2):
                        self.body.append(nn.Sequential(nn.Linear(width, width), ac_fn))
                    self.out = nn.Linear(width, 1)

                def forward(self, x):
                    x = self.input(x)
                    for layer in self.body:
                        x = layer(x)
                    res = self.out(x)
                    return res


            model = NeuralNetwork().to(device)
            model.apply(init_fn)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = np.array([])
            epochs = 15
            for t in range(epochs):
                #print(f"epoch {t + 1}", end=' ')
                epoch_losses = train(train_dataloader, model, loss_fn, optimizer)
                train_losses = np.append(train_losses, epoch_losses)

            fig, ax = plt.subplots()
            ax.plot(train_losses)
            ax.set_title(f"PyTorch: {depth}-depth, {width}-width network")
            ax.set_xlabel("Epoches")
            ax.set_ylabel("squared error")
            plt.savefig(f"torch_{ac_name}_depth_{depth}_width_{width}.png")
            plt.close()

            bank_train_df = pd.read_csv('bank-note/train.csv', header=None)
            bank_test_df = pd.read_csv('bank-note/test.csv', header=None)

            bank_train_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
            bank_test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

            X_train = bank_train_df.iloc[:, :-1].values
            Y_train = bank_train_df.iloc[:, -1].values
            X_test = bank_test_df.iloc[:, :-1].values
            Y_test = bank_test_df.iloc[:, -1].values

            torch_trainX = torch.from_numpy(X_train).float()
            torch_trainY = torch.from_numpy(Y_train).float()
            torch_textX = torch.from_numpy(X_test).float()
            torch_textY = torch.from_numpy(Y_test).float()

            print()
            train_error = test(model, torch_trainX, torch_trainY)
            print(f"Training error: {train_error:>8f} \n")
            test_error = test(model, torch_textX, torch_textY)
            print(f"Testing error: {test_error:>8f} \n")