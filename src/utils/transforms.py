from torch.nn import functional as nnF

class Normalization:
    def __call__(self, x):
        return (x.float() - x.mean()) / x.std()


class Resize:
    def __init__(self, to_size):
        self.to_size = to_size

    def __call__(self, x):
        return nnF.interpolate(
            x, self.to_size,
            align_corners = True,
            mode = "bilinear"
        )