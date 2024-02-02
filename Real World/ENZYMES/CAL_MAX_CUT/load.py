import CAL_MAX_CUT
import torch


def data_load(load_path):
    datas = torch.load(load_path)
    return datas
