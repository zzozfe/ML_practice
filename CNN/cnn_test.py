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

test_img = sys.argv[1]
predict_csv = sys.argv[2]

class cnn_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0])
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label

class cnn(nn.Module):
    
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),     
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(512, 7),
        )
        
    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) #(24,24)
        x = self.conv2(x) #(12,12)
        x = self.conv3(x) #(6,6)
        x = self.conv4(x) #(3,3)
        x = x.view(-1, 3*3*128)
        x = self.fc(x)
        return x    

model = cnn()
model_path = sys.argv[3]
model.load_state_dict(torch.load(model_path))

use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
model.eval()
    
test_img = sorted(os.listdir(test_img))
test_img = ['./test_img/' + i for i in test_img]
test_label = np.full(len(test_img),1)
test_data = list(zip(test_img, test_label))
test_data = cnn_dataset(test_data, transform)
test_set = DataLoader(test_data, batch_size=128, shuffle=False)
predict_value = []
for img, _ in test_set: 
    if use_gpu:
        img = img.cuda()
        _ = _.cuda()   
    output = model(img)
    predict_value += (torch.max(output, 1)[1]).tolist()
submission = pd.DataFrame()
submission['id'] = np.arange(len(test_img))
submission['label'] = predict_value
submission.to_csv(predict_csv, index = False)