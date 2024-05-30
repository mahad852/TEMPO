import os
import matplotlib.pyplot as plt


models = ["CustomLSTM", "CustomLSTM", "CustomLSTM", "CustomLSTM"]
seq_lens = [64, 3200, 128, 256]
forecast_lens = [64, 64, 64, 64]


res_files = [os.path.join("logs", "ecg_mit", f"{models[i]}_{seq_lens[i]}_{forecast_lens[i]}.txt") for i in range(len(seq_lens))]
identifier = "all_contexts"

def read_data(fpath):
    lines = []
    with open(fpath, "r") as f:
        lines = f.readlines()
        if len(lines) <= 1:
            raise Exception(f"File {fpath} must have more than 1 lines")
    
    rmses = []
    mses = []
    smapes = []
    maes = []

    pred_lens = []

    for line in lines[1:]:
        if len(line.split(";")) != 5:
            raise Exception(f"Each line in file {fpath} must be have 5 components separated by semi-colons")
        
        if "pred_len" not in line.split(";")[0].strip():
            raise Exception(f"First component in file {fpath} must be pred_len")
        
        pred_lens.append(int(line.split(";")[0].strip().split(":")[1].strip()))

        mses.append(float(line.split(";")[1].strip().split(":")[1].strip()))
        rmses.append(float(line.split(";")[2].strip().split(":")[1].strip()))
        maes.append(float(line.split(";")[3].strip().split(":")[1].strip()))
        smapes.append(float(line.split(";")[4].strip().split(":")[1].strip()))


    return pred_lens, mses, rmses, maes, smapes

def plot_graph(x, y, path, title, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot()

    for k in x.keys():
        ax.plot(x[k], y[k], label=f"Context {k}")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend(loc='best')
    
    fig.savefig(path)
    # plt.clf()

def plot_all_data(all_pred_lens, all_mses, all_rmses, all_maes, all_smapes, graph_identifier):
    root_dir = os.path.join("graphs", graph_identifier)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    plot_graph(all_pred_lens, all_mses, os.path.join(root_dir, "mse.png"), "MSE vs. forecast length", "Forecast Length", "Mean Square Error (MSE)")
    plot_graph(all_pred_lens, all_rmses, os.path.join(root_dir, "rmse.png"), "RMSE vs. forecast length", "Forecast Length", "Root Mean Square Error (RMSE)")
    plot_graph(all_pred_lens, all_maes, os.path.join(root_dir, "mae.png"), "MAE vs. forecast length", "Forecast Length", "Mean Absolute Error (MAE)")
    plot_graph(all_pred_lens, all_smapes, os.path.join(root_dir, "smape.png"), "SMAPE vs. forecast length", "Forecast Length", "Symmetric mean absolute percentage error (SMAPE)")


all_pred_lens, all_mses, all_rmses, all_maes, all_smapes = {}, {}, {}, {}, {}

for i in range(len(res_files)):
    res_file = res_files[i]
    context_len = seq_lens[i]

    pred_lens, mses, rmses, maes, smapes = read_data(res_file)

    all_pred_lens[context_len] = pred_lens
    all_mses[context_len] = mses
    all_rmses[context_len] = rmses
    all_maes[context_len] = maes
    all_smapes[context_len] = smapes

plot_all_data(all_pred_lens, all_mses, all_rmses, all_maes, all_smapes, identifier)