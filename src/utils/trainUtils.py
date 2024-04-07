from .losses import calc_ctc_loss
import numpy as np
import tqdm

def trainOneEpoch(model, optimizer, converter, device, ldr, epoch):
    losses = []  
    loop = tqdm.tqdm(ldr, colour = "blue")
    for (img, label) in loop:
        label, len_ = converter.encode(label)
        img = img.transpose(1, 2).to(device)

        pred = model(img)

        loss = calc_ctc_loss(pred, label, len_)
        losses.append(loss.cpu().detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _log = {
            "epoch" : epoch,
            "loss" : loss.item()
        }
        loop.set_postfix(_log)
    return np.mean(losses)