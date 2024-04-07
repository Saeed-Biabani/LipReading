from src.utils.labelConverter import CTCLabelConverter
from src.utils.transforms import Resize, Normalization
from src.utils.dataProvider import LipDataLoader
from src.utils.trainUtils import trainOneEpoch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from src.nn.model import LipNet
import config as cfg
import torch

def printConfigVars(module, fname):
    pa = [item for item in dir(module) if not (item.startswith("__") or item.endswith("_"))]
    for item in pa:
        value = eval(f'{fname}.{item}')
        if str(type(value)) not in ("<class 'module'>", "<class 'function'>"):
            print(f"{fname}.{item} : {eval(f'{fname}.{item}')}")

printConfigVars(cfg, 'cfg')

device = cfg.device

trainds = LipDataLoader(
    root = cfg.root,
    sub_bch = cfg.ts_len,
    transforms = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        Normalization()
    ])
); trian_dataloader = DataLoader(trainds, cfg.batch_size, True)

model = LipNet(len(cfg.vocab)).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = cfg.learning_rate,
    amsgrad = True
)
converter = CTCLabelConverter(
    dict_=cfg.vocab,
    max_str_len = cfg.max_len,
    device = device
)

train_loss = []
for epoch in range(1, cfg.epochs+1):
    _loss = trainOneEpoch(
        model,
        optimizer,
        converter,
        device,
        trian_dataloader,
        epoch
    )
    train_loss.append(_loss)
torch.save(model.state_dict(), "LipReader.pth")