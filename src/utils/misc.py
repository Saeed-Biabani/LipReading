import matplotlib.pyplot as plt
import random
import numpy as np

def plotSamples(ds):
    indx  = random.randint(0, len(ds) - 1)
    ts, label = ds[indx]

    print(ds.dirs_[indx])

    plt.figure(figsize = (5, 18))
    plt.suptitle(label.title(), fontsize = 30)

    for i in range(22):
        plt.subplot(11, 2, i+1)
        plt.title(i+1)
        img = np.moveaxis(ts[i].cpu().numpy(), 0, -1)
        plt.imshow(img, cmap = "gray")
        plt.axis("off")
    plt.savefig(f"{label.title()}.png")


def printParams(model, text):
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    print(text.format(sum(params_num)))


def plotHistory(list_):
    plt.figure(figsize = (10, 5))
    plt.title("Learning Curve")
    plt.plot(list_, 'green')
    plt.ylabel("CTC Loss")
    plt.xlabel("Epoch")
    plt.legend(["train loss"], loc = "upper right")
    plt.savefig("learningCurve.png")
    plt.close()