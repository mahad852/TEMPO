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
import wfdb
from statsmodels.tsa.seasonal import STL
import time

warnings.filterwarnings('ignore')

stl_position = 'stl/'

class Dataset_ECG_MIT(Dataset):
    def __init__(self, root_path, data_path = '', flag='train', size=None,
                 features='S', target='OT', scale=True, timeenc=0, freq='ms', 
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
        self.data_name = data_name
        save_stl = stl_position + self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                res = STL(df, period = 250).fit()

                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp


    def __read_data__(self):
        self.scaler = StandardScaler()
        
        total = 0
        for file in os.listdir(self.root_path):
            if len(file.split('.')) > 1 and file.split('.')[-1] == 'dat':
                total += 1

        cols = []
        data = np.zeros(shape = (650000, total * 2))

        i = 0

        for file in os.listdir(self.root_path):
            if len(file.split('.')) > 1 and file.split('.')[-1] == 'dat':
                fname = file.split('.')[0]
                record = wfdb.rdrecord(os.path.join(self.root_path, fname))
                data[:, i] = record.__dict__["p_signal"][:, 0]
                data[:, i + 1] = record.__dict__["p_signal"][:, 1]
                
                cols.extend(list(map(lambda s : f"{fname}_{s}", record.__dict__["sig_name"])))

                i += 2
        
        df_raw = pd.DataFrame(data, columns=cols)

        # border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        df_data = df_raw

        if self.scale:
            train_data = df_data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        curr_time = time.time() * 1000
        time_step = (1/360) * 1000
        data_stamp = pd.to_datetime([curr_time + (i * time_step) for i in range(1, 650001)], unit='ms').values

        
        # After we get data, we do the stl resolve
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=df_raw, data_name=self.data_name)
        # end -dove

        # if self.timeenc == 1:
        #     data_stamp = time_features(data_stamp, freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)

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
