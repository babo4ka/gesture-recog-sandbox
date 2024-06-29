import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

train_path = "../sign_dataset/sign_mnist_train/sign_mnist_train.csv"
test_path = "../sign_dataset/sign_mnist_test/sign_mnist_test.csv"

traindf = pd.read_csv(train_path)
testdf = pd.read_csv(test_path)


def set_dataset(df):
    x_data = []
    y_data = []

    for i, row in df.iterrows():
        y_data.append(row['label'])

        x_temp = row.drop('label')#/255

        nparr = np.array(x_temp)
        t = torch.tensor(nparr)
        t = torch.reshape(t, (28, 28)).float()
        x_data.append(t)

    y_data = torch.tensor(y_data)

    x_data = np.array(x_data)
    x_data = torch.tensor(x_data)
    x_data = x_data.unsqueeze(1).float()
    return x_data, y_data


x_train, y_train = set_dataset(traindf)
print(x_train.shape)

x_test, y_test = set_dataset(testdf)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(0.5)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(84, 25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)



        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        #print(x.size(0), x.size(1), x.size(2), x.size(3))

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.drop(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


net = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=1.0e-3)
sh = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)

batch_size = 100

test_accuracy_history = []
test_loss_history = []

train_loss_history = []

x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)

accuracy_max = 0

for epoch in range(50):
    order = np.random.permutation(len(x_train))

    for start_index in range(0, len(x_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = x_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        preds = net.forward(X_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    sh.step()

    test_preds = net.forward(x_test)
    test_loss_history.append(loss(test_preds, y_test).data.cpu())

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
    test_accuracy_history.append(accuracy)
    if accuracy > accuracy_max:
        torch.save(net, "GestRecogNet.pt")
        accuracy_max = accuracy
    print(accuracy)

    train_preds = net.forward(x_train)
    train_loss_history.append(loss(train_preds, y_train).data.cpu())


print("max accuracy: {0:.2f}".format(accuracy_max))
plt.axhline(y=0.992, color='r', linestyle='-', label='accuracy 0.992')
plt.plot(test_accuracy_history, label='accuracy')

plt.plot(test_loss_history, label='loss validation')

plt.plot(train_loss_history, label='loss train')

plt.legend()

plt.show()
