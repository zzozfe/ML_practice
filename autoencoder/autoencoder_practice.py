from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys

train_path = sys.argv[1]
predict_path = sys.argv[2]
model_path = sys.argv[3]

def load_data(path):
    x = np.load(path)
    x = (x / 255.0)*2-1
    return x   

class autoencoder_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        label = idx
        return img, label
    def __getimg__(self, idx):
        img = self.__getitem__(idx)[0]
        img = img.numpy().T
        img = ((img+1)/2)*255
        img = img.astype(np.uint8) 
        return Image.fromarray(img)   
    
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                    #input (3,32,32)
            nn.Conv2d(3, 64, 5, stride=2, padding=1),   #(8,16,16)   
            nn.LeakyReLU(0.5),                     
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  #(16,8,8)
            nn.LeakyReLU(0.5),  
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3,2,1),                   #(32,4,4)
            nn.LeakyReLU(0.5),
            nn.BatchNorm2d(256),    
        )
        self.fc1 = nn.Linear(256*4*4,32)
        self.fc2 = nn.Linear(32,256*4*4)

        self.decoder = nn.Sequential(                    #input(32,2,2)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),          #(32,8,8)
            nn.LeakyReLU(0.5),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),         #output(16,8,8)
            nn.LeakyReLU(0.5),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        xe = self.encoder(x)
        xe = xe.view(len(xe),-1)
        xe = self.fc1(xe)
        xd = self.decoder(self.fc2(xe).view(-1,256,4,4))
        return xe, xd

use_gpu = torch.cuda.is_available()

train_X = load_data(train_path)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = autoencoder_dataset(train_X, transform)
train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True)

model = autoencoder()
model.load_state_dict(torch.load(model_path))
model.double()
if use_gpu:
    model = model.cuda()

    
def latent(x):
    x = x.cpu().detach().numpy()
    return x

test_loader = DataLoader(train_dataset, batch_size = 32)
predict = []
latentss = []
outputs = []
for img,_ in test_loader:
    if use_gpu:
        img = img.cuda()
    encoder, output = model(img)
    outputs.append(latent(output))
    latentss.append(latent(encoder))
    predict += torch.max(output, 1)[1].tolist()
latentss = np.concatenate(latentss, axis=0)
latents = latentss.reshape([9000,-1])  
latents_mean = np.mean(latents, axis=1).reshape(9000,-1)
latents_std = np.std(latents, axis=1).reshape(9000,-1)
latents = (latents - latents_mean)/latents_std

from sklearn import manifold

tsne = manifold.TSNE()
tsne_x = tsne.fit_transform(latents)

from sklearn.cluster import KMeans

result = KMeans(n_clusters=2).fit(tsne_x).labels_

submission = pd.DataFrame()
ids = np.arange(9000)
submission['id'] = ids
submission['label'] = result.reshape(9000,-1)

if submission['label'][1] == 1:
    submission['label'] = submission['label'].map(lambda x : abs(x-1))

submission.to_csv(predict_path, index =False)