import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim.lr_scheduler
import os
from tqdm import tqdm_notebook as tqdm
import cv2

def readfile(path, label):

    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in tqdm(enumerate(image_dir)):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] =cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])

    if label:
        return x, y
    else:
        return x

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):   
        self.x = x

        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


class new_resnet18(nn.Module):
    def __init__(self):
        super(new_resnet18, self).__init__()

        self.model = models.resnet18(pretrained=False)

        self.model.fc = nn.Linear(512, 39)

    def forward(self, x):
        out = self.model(x)
        return out