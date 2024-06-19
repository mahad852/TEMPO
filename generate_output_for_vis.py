import numpy as np
import torch
from models.CustomLSTM import CustomLSTM
import os

dataset_path = "/Users/ma649596/Downloads/MIT-BIH_lagllama_384_64_forecast.npz"

sample_key = "a103-48_384_64_"
sample_true_key = f"{sample_key}true"

context_len = 384
pred_len = 64

def get_context(ds, true_key):
    return ds[true_key][:context_len]

model_checkpoint = "lora_revin_6domain_checkpoints_1/ECG_MIT_TEMPO_6_prompt_learn_384_64_100_sl336_ll168_pl64_dm768_nh4_el3_gl6_df768_ebtimeF_itr0/checkpoint.pth"

dataset = np.load(dataset_path)
device = torch.device("cpu")

model = CustomLSTM(context_len, pred_len)
model.load_state_dict(torch.load(model_checkpoint), strict=False)

context = torch.tensor(np.array([get_context(dataset, sample_true_key)]), device=device)
forecast = model(context).detach().numpy()[0].squeeze(-1)

if not os.path.exists("forecasts"):
    os.mkdir("forecasts")

np.save("forecasts/a103-48_384_64_lstm", forecast)