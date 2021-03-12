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

train_img = sys.argv[1]
train_label = sys.argv[2]

def load_data(img_path, label_path):
    train_image = sorted(os.listdir(img_path))
    train_image = ['./train_img/' + i for i in train_image]
    train_label = pd.read_csv(label_path)
    train_label = train_label['label'].values.tolist()
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    train_set = train_data[:26000]
    val_set = train_data[26000:]
    
    return train_data, val_set

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
    
if __name__ == '__main__': 
    
    use_gpu = torch.cuda.is_available()
    
    train_set, val_set = load_data(train_img, train_label)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = cnn_dataset(train_set, transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    val_dataset = cnn_dataset(val_set, transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = cnn()
    if use_gpu:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    loss_fn = nn.CrossEntropyLoss()
    
    num_epoch = 15
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(val_loader):
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
        
        if np.mean(train_acc) > 0.9:
            checkpoint_path = 'model_{}.pkl'.format(epoch+1) 
            torch.save(model, checkpoint_path)
            print('model saved to %s' % checkpoint_path)
            