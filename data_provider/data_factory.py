from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_ECG_chapman
from data_provider.ecg_mit import Dataset_ECG_MIT
from torch.utils.data import DataLoader, WeightedRandomSampler

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'ecg_chapman' : Dataset_ECG_chapman,
    'ecg_mit' : Dataset_ECG_MIT
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all,
        data_name = args.data_name
    )
    print(flag, len(data_set))

    if args.data == "ecg_chapman" and flag == "train":
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=args.num_workers,
            sampler=WeightedRandomSampler(data_set.sample_weights, len(data_set.sample_weights)),
            drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
