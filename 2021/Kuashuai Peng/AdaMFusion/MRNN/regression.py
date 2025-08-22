import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('MRNN_training_data.txt')

# Extract features and labels
X = df.drop(['w_0', 'accuracy', 'precision', 'recall', 'f1-score'], axis=1)
y = df['accuracy']

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing data
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the model
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 6),
    nn.ReLU(),
    nn.Linear(6, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

n_epochs = 100  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
plt.plot(history)
plt.show()

centre_weight = (0.0, 0.254, 0.343, 0.169, 0.234)
X_w1 = [[round(w_1, 3), centre_weight[2], centre_weight[3], centre_weight[4]] for w_1 in np.linspace(0, 1, 1000)]
X_w2 = [[centre_weight[1], round(w_2, 3), centre_weight[3], centre_weight[4]] for w_2 in np.linspace(0, 1, 1000)]
X_w3 = [[centre_weight[1], centre_weight[2], round(w_3, 3), centre_weight[4]] for w_3 in np.linspace(0, 1, 1000)]
X_w4 = [[centre_weight[1], centre_weight[2], centre_weight[3], round(w_4, 3)] for w_4 in np.linspace(0, 1, 1000)]
X_w = (X_w1, X_w2, X_w3, X_w4)

model.eval()
with torch.no_grad():
    for i, X in enumerate(X_w):
        X_scaled = X_scaler.transform(X)
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
        y_pred = model(X_scaled)
        X = [x[i] for x in X]
        y = y_scaler.inverse_transform(y_pred) * 100
        plt.plot(X, y)
        plt.xlabel(f'weighting factor w{i + 1}')
        plt.ylabel('Accuracy(%)')
        plt.savefig(f'weighting_factor_w{i + 1}.png')
        plt.show()
