import torch

class HyperParams:
    IMAGE_H, IMAGE_W = 40, 40
    N_ITERS = 100
    LR = 0.01
    BATCH_SIZE = 64
    N_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")