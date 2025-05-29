import torch
from torch import nn, optim, arange
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

df = pd.read_csv("/content/Salary_dataset.csv")
df.describe()
X = df['YearsExperience'].to_numpy().reshape(-1,1)
y = df['Salary'].to_numpy().reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)
X_scale = StandardScaler()
y_scale = StandardScaler()

X_train_scale = torch.tensor(X_scale.fit_transform(X_train), dtype=torch.float32)
X_test_scale = torch.tensor(X_scale.transform(X_test), dtype=torch.float32)

y_train_scale = torch.tensor(y_scale.fit_transform(y_train), dtype=torch.float32)
y_test_scale = torch.tensor(y_scale.transform(y_test), dtype=torch.float32)


fig, (ax1, ax2) =  plt.subplots(1,2 , figsize=(10,4))


def plot_data(X_train = X_train_scale,
              X_test = X_test_scale,
              y_train = y_train_scale,
              y_test = y_test_scale,
              prediction= None,
              plot = ax1
              ):
  # ax1.figure(figsize=(8,5))

  ax1.scatter(X_train, y_train, c='g')
  ax1.scatter(X_test, y_test, c='b')

  if prediction is not None:
    ax1.scatter(X_test, prediction, c='r')

  ax1.legend(['Train', 'Test', 'Prediction'])


class LinearRegression(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer = nn.Linear(1,1)

  def forward (self, x):
    return self.layer(x)

def accuracy (y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_true)) * 100
  return acc


model_1 = LinearRegression()
loss_function = nn.L1Loss()
opt = optim.SGD(params=model_1.parameters(), lr =0.01)

epochs = 250

train_loss_val = []
test_loss_val = []
numb_epoch = []

for epoch in arange(epochs):
  model_1.train()

  y_train_pred = model_1(X_train_scale)

  loss = loss_function(y_train_pred, y_train_scale)
  opt.zero_grad()

  loss.backward()
  opt.step()

  model_1.eval()

  y_test_pred = model_1(X_test_scale)
  if epoch % 10 == 0:
    numb_epoch.append(epoch)
    test_loss = loss_function(y_test_pred, y_test_scale)
    test_loss_val.append(test_loss.item())
    train_loss_val.append(loss.item())

new_pred = model_1(X_test_scale)

print(new_pred.detach().numpy())


with torch.inference_mode():

  plot_data(prediction=new_pred)
  ax2.plot(numb_epoch, test_loss_val)
  ax2.plot(numb_epoch, train_loss_val)
  ax2.legend(['Test loss', 'Train loss'])
