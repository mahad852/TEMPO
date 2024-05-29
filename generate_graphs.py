import os
import matplotlib.pyplot as plt


model = "CustomLSTM"
seq_len = 64
forecast_len = 64

res_file = os.path.join("logs", "ecg_mit", f"{model}_{seq_len}_{forecast_len}.txt")
identifier = f"{model}_{seq_len}_{forecast_len}"

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

    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    fig.savefig(path)
    # plt.clf()

def plot_all_data(pred_lens, mses, rmses, maes, smapes, graph_identifier):
    root_dir = os.path.join("graphs", graph_identifier)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    plot_graph(pred_lens, mses, os.path.join(root_dir, "mse.png"), "MSE vs. forecast length", "Forecast Length", "Mean Square Error (MSE)")
    plot_graph(pred_lens, rmses, os.path.join(root_dir, "rmse.png"), "RMSE vs. forecast length", "Forecast Length", "Root Mean Square Error (RMSE)")
    plot_graph(pred_lens, maes, os.path.join(root_dir, "mae.png"), "MAE vs. forecast length", "Forecast Length", "Mean Absolute Error (MAE)")
    plot_graph(pred_lens, smapes, os.path.join(root_dir, "smape.png"), "SMAPE vs. forecast length", "Forecast Length", "Symmetric mean absolute percentage error (SMAPE)")

pred_lens, mses, rmses, maes, smapes = read_data(res_file)

plot_all_data(pred_lens, mses, rmses, maes, smapes, identifier)