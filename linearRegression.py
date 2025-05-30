import torch
from torch import nn, optim, arange
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv("/content/Salary_dataset.csv")
torch.manual_seed(42)

X = df['YearsExperience'].to_numpy().reshape(-1,1)
y = df['Salary'].to_numpy().reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_scale = StandardScaler()
y_scale = StandardScaler()

X_train_scale = torch.tensor(X_scale.fit_transform(X_train), dtype=torch.float32)
X_test_scale = torch.tensor(X_scale.transform(X_test), dtype=torch.float32)

y_train_scale = torch.tensor(y_scale.fit_transform(y_train), dtype = torch.float32)
y_test_scale = torch.tensor(y_scale.transform(y_test), dtype=torch.float32)

figure, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

def plot_prediction (X_train = X_train,
                     X_test = X_test,
                     y_train = y_train,
                     y_test = y_test,
                     prediction = None,
                     plot = ax1
                     ):
  

  plot.set_title('Train, Test and prediction plot')
  plot.set_xlabel("Years of Experience")
  plot.set_ylabel('Salary')

  plot.scatter(X_train, y_train, c= 'b', label="train data")
  plot.scatter(X_test, y_test, c='g', label='test data')

  if prediction is not None:
    plot.scatter(X_test, prediction, c='r', label="prediction")

  plot.legend()

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer = nn.Linear(1,1)

  def forward(self, x):
    return self.layer(x)

model_1 = LinearRegressionModel()

loss_function = nn.MSELoss()
opt = optim.SGD(params=model_1.parameters(), lr=0.01)


epochs = 500
train_loss_vals =[]
test_loss_vals =[]
epoch_number = []
r2_score_vals =[]



for epoch in arange(epochs):
  model_1.train()

  y_train_pred = model_1(X_train_scale)
  train_loss = loss_function(y_train_pred, y_train_scale)

  opt.zero_grad()
  train_loss.backward()

  opt.step()

  model_1.eval()

  with torch.inference_mode():
    y_test_pred = model_1(X_test_scale)
    y_test_pred_unscale = y_scale.inverse_transform(y_test_pred.detach().numpy())
    if epoch % 5 == 0:
      test_loss = loss_function(y_test_pred, y_test_scale)
      train_loss_vals.append(train_loss.item())
      test_loss_vals.append(test_loss.item())
      epoch_number.append(epoch)
      r2_score_vals.append(r2_score(y_test_scale, y_test_pred))

y_final_pred = model_1(X_test_scale)
y_final_pred_unscale = y_scale.inverse_transform(y_final_pred.detach().numpy())
plot_prediction(prediction=y_final_pred_unscale)

ax2.plot(epoch_number, train_loss_vals, label='train loss')
ax2.plot(epoch_number, test_loss_vals, label='test loss')
ax2.plot(epoch_number, r2_score_vals, label='R2 score')
ax2.set_xlabel ('Number of epoch')
ax2.set_ylabel("loss ")
ax2.legend()
