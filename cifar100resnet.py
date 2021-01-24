import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train = unpickle('data/train')
X_train = train[b'data']
X_train = X_train.reshape(50000, 3, 32, 32)
X_train = X_train/255
X_train = torch.from_numpy(X_train.astype('float32'))
y_train = train[b'fine_labels']
y_train = torch.tensor(y_train)



test = unpickle('data/test')
X_test = test[b'data']
X_test = X_test.reshape(10000, 3, 32, 32)
X_test = X_test/255
X_test = torch.from_numpy(X_test.astype('float32'))
y_test = test[b'fine_labels']
y_test = torch.tensor(y_test)

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0)

dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=0)

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), padding=1)
    def forward(self, x_in):
        x = F.gelu(self.conv1(x_in)) #zamiast relu sprobowac uzyc gelu
        x = F.gelu(self.conv2(x))
        return x + x_in



class NeuralNetwork(pl.core.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(3, 3))
        self.res1 = ResidualBlock(20)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.res2 = ResidualBlock(20)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.res3 = ResidualBlock(20)
        #trzy rsbloki a pomiedzy warstwy poolingowe

        # self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3, 3))
        # self.conv3 = nn.Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3))
        # self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.dense1 = nn.Linear(in_features=980, out_features=1024)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dropout2 = nn.Dropout(0.25)
        self.dense3 = nn.Linear(in_features=256, out_features=100)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = x.view(-1, 980)
        x = F.glu(self.dense1(x))
        x = self.dropout1(x)
         #dodac dropout po warstwach dense,zzamiast relu to glu
        x = F.glu(self.dense2(x)) 
        x = self.dropout2(x)
        x = F.log_softmax(self.dense3(x))
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        self.log('train_accuracy',pl.metrics.functional.accuracy(y_pred, y))
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        self.log('val_accuracy', pl.metrics.functional.accuracy(y_pred, y))
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        self.log('test_accuracy',pl.metrics.functional.accuracy(y_pred, y))
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


neural_network = NeuralNetwork()
#X, y = next(iter(dataset_train))
#neural_network(X).shape
trainer = pl.Trainer(max_epochs=10, gpus=-1)
trainer.fit(neural_network, dataset_train, dataset_test)

trainer.test(neural_network, dataset_test)

#overfittowal, zmniejszyc liczbe neuronow? dodac dropout