import numpy as np
import torch
import os


ts_len = 22
root = "../../../ForFun/LipDS/best-lip-reading-dataset/outputs"
dirs_ = list(os.listdir(root))
words_ = [x.split('_')[0] for x in dirs_]
vocab = ''.join(np.unique(list(''.join(words_))))
max_len = len(max(words_, key = len))
device = torch.device("cuda")
batch_size = 16
img_h = 64
img_w = 128
img_channel = 1
learning_rate = 2e-5
epochs = 400