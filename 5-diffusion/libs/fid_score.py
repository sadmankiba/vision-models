import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from PIL import Image
from torchvision import transforms
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader, Dataset



class InceptionV3Feats(nn.Module):
    def __init__(self):
        super(InceptionV3Feats, self).__init__()
        model = inception_v3(weights="DEFAULT")
        
        self.blocks = nn.ModuleList()
        block0 = [
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        block1 = [
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block1))

        block2 = [
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))

        block3 = [
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ]
        self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = 2 * x - 1  # From range (0, 1) to range (-1, 1)

        for block in self.blocks:
            x = block(x)

        return x

class ImageDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    

def calc_inception_mu_sigma(model, path, device):
    path = pathlib.Path(path)
    img_exts = {"jpg", "jpeg", "png"}
    dims = 2048
    batch_size = 4
    files = [file for ext in img_exts for file in path.glob("*.{}".format(ext))]
    
    model.eval()
    
    dataset = ImageDataset(files, transforms=transforms.ToTensor())
    dataloader = DataLoader(dataset,batch_size=batch_size)

    pred_features = np.empty((len(files), dims))
    start_idx = 0
    for batch in dataloader:
        batch = batch.to(device) # (batch_size, 3, img_h, img_w)

        with torch.no_grad():
            pred = model(batch)   # (batch_size, dims, 1, 1)

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_features[start_idx : start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_features, axis=0)
    sigma = np.cov(pred_features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, mu2, sigma1, sigma2):
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_fid_score(path1, path2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3Feats().to(device)

    m1, s1 = calc_inception_mu_sigma(model, path1, device)
    m2, s2 = calc_inception_mu_sigma(model, path2, device)
    fid_value = frechet_distance(m1, m2, s1, s2)

    return fid_value