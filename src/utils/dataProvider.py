from torch.utils.data import Dataset
from torchvision import io
import pathlib
import torch
import os

class LipDataLoader(Dataset):
    def __init__(self, root, transforms = None, sub_bch = 22):
        super(LipDataLoader, self).__init__()
        self.root = root
        self.sub_bch = sub_bch
        self.transforms = transforms
        self.dirs_ = list(os.listdir(root))


    def __len__(self):
        return len(self.dirs_)


    def __extr_label__(self, s):
        return s.split('/')[-1].split('_')[0]


    def __load_images__(self, gen):
        ts = torch.zeros((self.sub_bch, 1, 80, 112))
        for i, fname in enumerate(sorted(gen, key = lambda x : int(x.stem))):
            ts[i] = io.read_image(
                str(fname),
                mode = io.ImageReadMode.GRAY
            )
        return ts


    def __getitem__(self, indx):
        path_ = os.path.join(self.root, self.dirs_[indx])
        list_ = pathlib.Path(path_).glob("*.png")

        ts = self.__load_images__(list_)
        label = self.__extr_label__(path_)

        if self.transforms != None:
            ts = self.transforms(ts)

        return ts, label