import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import torch
import torch.nn as nn

from models.CustomLSTM import CustomLSTM


context_len = 720
pred_len = 1

# ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")
data_path = "/home/user/MIT-BIH_720_1_many.npz"
model_checkpoint = "lstm_512_64/checkpoint.pth"

def single_loader(dataset):
    for i in range(len(dataset.files)):
        xy = dataset[dataset.files[i]]
        x, y = xy[:context_len], xy[context_len:context_len + pred_len]

        yield [x], [y]

dataset = np.load(data_path)
device = torch.device("cuda")

model = CustomLSTM(context_len, pred_len)
model.load_state_dict(torch.load(model_checkpoint), strict=False)
model = model.to(device=device)

model.eval()


mses = []
maes = []

mse_by_pred_len = {}
rmse_by_pred_len = {}
mae_by_pred_len = {}

total = 0

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] = 0.0
    rmse_by_pred_len[p_len] = 0.0
    mae_by_pred_len[p_len] = 0.0

for i, (x, y) in enumerate(single_loader(dataset)):
    forecast = model(torch.tensor(np.array(x), device=device).unsqueeze(-1))[:, -pred_len:, :]
    
    y = torch.tensor(y).unsqueeze(-1)[:, -pred_len:, :].to(device=device)

    mse = nn.functional.mse_loss(y, forecast).item()
    rmse = np.sqrt(mse)
    mae = nn.functional.l1_loss(y, forecast).item()

    mses.append(mse)
    maes.append(mae)

    total += 1

    for p_len in range(1, pred_len + 1):
        mse_by_pred_len[p_len] += nn.functional.mse_loss(y[:, :p_len], forecast[:, :p_len]).item()
        mae_by_pred_len[p_len] += nn.functional.l1_loss(y[:, :p_len], forecast[:, :p_len]).item()

    if i % 1000 == 0:
        print(f"iteraition: {i} | MSE: {mse} RMSE: {rmse} MAE: {mae}")

print(f"MSE: {np.average(mses)} RMSE: {np.sqrt(np.average(mses))} MAE: {np.average(maes)}")

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] /= total
    rmse_by_pred_len[p_len] = np.sqrt(mse_by_pred_len[p_len])
    mae_by_pred_len[p_len] /= total

if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"LSTM_{context_len}_{pred_len}_split.csv"), "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_pred_len[p_len]},{rmse_by_pred_len[p_len]},{mae_by_pred_len[p_len]}")
        if p_len != pred_len:
            f.write("\n")