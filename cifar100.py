import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

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

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3, 3), padding=0)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3), padding=0)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=0)
        self.dense1 = nn.Linear(in_features=11520, out_features=2048)
        self.dense2 = nn.Linear(in_features=2048, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 11520)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.log_softmax(self.dense3(x))
        return x


model = NeuralNetwork()
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters())

for i in range(10):
    for X, y in dataset_train:
        optimizer.zero_grad()
        X = X.to('cuda')
        y = y.to('cuda')
        y_pred = model(X)
        loss = F.nll_loss(y_pred, y)
        loss.backward()
        optimizer.step()
    print(i)

X_test = X_test.to('cpu')
y_test = y_test.to('cpu')
model = model.to('cpu')
y_pred = model(X_test)
y_pred = torch.exp(y_pred)
y_pred = torch.argmax(y_pred, dim=1)
y_pred.shape
accuracy_score(y_test.detach().numpy(), y_pred.detach().numpy())