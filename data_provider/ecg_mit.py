import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
import pickle
from statsmodels.tsa.seasonal import STL
import time
from scipy import signal
from scipy.signal import medfilt

warnings.filterwarnings('ignore')

stl_position = 'stl/'

def moving_average(a, n=3):
    a = np.pad(a, (1, 1), 'edge')
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def notch_filter(a):
    samp_freq = 360.0  # Sample frequency (Hz)
    notch_freq = 60.0  # Frequency to be removed from signal (Hz)
    quality_factor = 2  # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    return signal.filtfilt(b_notch, a_notch, a)

def normalize(a):
    return (a - np.min(a))/(np.max(a) - np.min(a))

def preprocess(a):
    a = moving_average(a)
    a = medfilt(a, kernel_size=3)
    a = notch_filter(a)
    return normalize(a)    

class Dataset_ECG_MIT(Dataset):
    def __init__(self, root_path, data_path = '', flag='train', size=None,
                 features='S', target='OT', scale=False, timeenc=0, freq='ms', 
                 percent=100, data_name = 'ecg_mit', max_len=-1, train_all=False):
        # info
        if size == None:
            self.seq_len = 250 * 4 * 4
            self.label_len = 250 * 4
            self.pred_len = 250 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        if self.freq == "ms":
            self.freq = f"{str(round((1/360) * 1000, 6))}ms"

        self.root_path = root_path
        self.data_name = data_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
            
    def stl_resolve(self, data_raw, data_name):
        """
        STL Global Decomposition
        """
        # self.data_name = 'etth1'

        [_,m] = data_raw.shape

        trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
        seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
        resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
        return trend_stamp, seasonal_stamp, resid_stamp


    def __read_data__(self):
        self.scaler = StandardScaler()
        
        cols = []
        data = None
        use_split_data = True

        if self.pred_len + self.seq_len == 721 and use_split_data:
            with np.load(os.path.join(self.root_path, "MIT-BIH_720_1_many.npz")) as d:
                data = np.zeros(shape = (721, len(d.files)))
                cols = d.files
                for i, file in enumerate(d.files):
                    data[:, i] = d[file]
        else:
            with np.load(os.path.join(self.root_path, "MIT-BIH.npz")) as d:
                data = np.zeros(shape = (650000, len(d.files)))
                cols = d.files
                for i, file in enumerate(d.files):
                    data[:, i] = d[file]

        df_raw = pd.DataFrame(data, columns=cols)

        train_percentage = 0.30

        num_columns = df_raw.shape[1]
        num_rows = df_raw.shape[0]

        if self.set_type == 0:
            border1, border2 = 0, int(num_columns * train_percentage)
            # border1, border2 = 0, int(num_rows * train_percentage)
        elif self.set_type == 1:
            border1, border2 = int(num_columns * train_percentage), num_columns
            # border1, border2 = int(num_rows * train_percentage), num_rows
        elif self.set_type == 2:
            border1, border2 = int(num_columns * train_percentage), num_columns
            # border1, border2 = int(num_rows * train_percentage), num_rows
        
        # df_data = df_raw.iloc[border1:border2, :]
        df_data = df_raw.iloc[:, border1:border2]

        data = df_data.values

        curr_time = time.time() * 1000
        time_step = (1/360) * 1000
        data_stamp = torch.tensor([curr_time + (i * time_step) for i in range(1, 650001)])
        
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=df_raw, data_name=self.data_name)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

        self.trend_stamp = trend_stamp
        self.seasonal_stamp = seasonal_stamp
        self.resid_stamp = resid_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
