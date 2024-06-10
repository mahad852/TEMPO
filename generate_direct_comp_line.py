import os
import matplotlib.pyplot as plt


models = ["CustomLSTM", "CustomLSTM", "CustomLSTM", "CustomLSTM", "CustomLSTM", "CustomLSTM", "CustomLSTM", "CustomLSTM"]
seq_lens = [64, 128, 256, 384, 512, 640, 1600, 3200]
forecast_lens = [64, 64, 64, 64, 64, 64, 64, 64]


res_files = [os.path.join("logs", "ecg_mit", f"{models[i]}_{seq_lens[i]}_{forecast_lens[i]}.txt") for i in range(len(seq_lens))]
identifier = "all_line"

def read_data(fpath):
    lines = []
    with open(fpath, "r") as f:
        lines = f.readlines()
        if len(lines) <= 1:
            raise Exception(f"File {fpath} must have more than 1 lines")

    line = lines[0]
    if len(line.split(";")) != 7:
        raise Exception(f"First line in file {fpath} must have 7 components separated by semi-colons")
    
    if "Epoch" not in line.split(";")[0].strip():
        raise Exception(f"First component in file {fpath} must be Epoch")
        
        
    mse = float(line.split(";")[4].strip().split(":")[1].strip())
    rmse = float(line.split(";")[5].strip().split(":")[1].strip())
    mae = float(line.split(";")[6].strip().split(":")[1].strip())
    # smapes.append(float(line.split(";")[4].strip().split(":")[1].strip()))

    return mse, rmse, mae

def plot_graph(dict, path, title, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot()

    x = []
    y = []
    for k in dict.keys():
        x.append(k)
        y.append(dict[k])

    ax.plot(x, y, marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # ax.legend(loc='best')
    
    fig.savefig(path)
    # plt.clf()

def plot_all_data(all_mses, all_rmses, all_maes, graph_identifier):
    root_dir = os.path.join("graphs", graph_identifier)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    plot_graph(all_mses, os.path.join(root_dir, "mse.png"), "MSE vs. Context Length", "Context Length", "Mean Square Error (MSE)")
    plot_graph(all_rmses, os.path.join(root_dir, "rmse.png"), "RMSE vs. Context Length", "Context Length", "Root Mean Square Error (RMSE)")
    plot_graph(all_maes, os.path.join(root_dir, "mae.png"), "MAE vs. Context Length", "Context Length", "Mean Absolute Error (MAE)")

all_mses, all_rmses, all_maes = {}, {}, {}

for i in range(len(res_files)):
    res_file = res_files[i]
    context_len = seq_lens[i]

    mse, rmse, mae = read_data(res_file)
    
    all_mses[context_len] = mse
    all_rmses[context_len] = rmse
    all_maes[context_len] = mae

plot_all_data(all_mses, all_rmses, all_maes, identifier)