import numpy as np
import pandas as pd
import spacy
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from gensim.models import word2vec
import sys
train_path = sys[1]
trainY_path = sys[2]
nlp = spacy.load('en_core_web_md')
tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)
def read_data(x):
    stopword = [',','.']
    for i in stopword:
        x['comment'] = x['comment'].map(lambda x : x.replace(i,''))
    ptt = []
    for i in x['comment'].values:
        ptt.append(nlp(i)) 
    ptt_remove_stop = []
    for doc in ptt:
        ptt_remove_stop.append([token.lemma_ for token in doc if not token.is_stop])
    model_w2v = word2vec.Word2Vec(ptt_remove_stop, size = 100)    
    w2v = []
    for _, key in enumerate(model_w2v.wv.vocab):
        w2v.append((key, model_w2v.wv[key]))
    special_tokens = ["<PAD>", "<UNK>"]
    for token in special_tokens:
        w2v.append((token, [0.0] * 100)) 
    return  w2v,ptt_remove_stop
class Vocab:
    def __init__(self, w2v, data):
        self.w2v = w2v
        self.data = data
        self._idx2token = [token for token, _ in w2v]
        self._token2idx = {token: idx for idx,
                           token in enumerate(self._idx2token)}
        self.PAD, self.UNK = self._token2idx["<PAD>"], self._token2idx["<UNK>"]

    def trim_pad(self, tokens, seq_len):
        return tokens[:min(seq_len, len(tokens))] + [self.PAD] * (seq_len - len(tokens))

    def convert_tokens_to_indices(self, tokens):
        return [
            self._token2idx[token]
            if token in self._token2idx else self.UNK
            for token in tokens]
    
    def data2v(self, idx):
        indx = self.convert_tokens_to_indices(self.data[idx])
        indx = self.trim_pad(indx, 30)
        indx = [self.w2v[token][1] for token in indx]
        return np.array(indx)
    
    def data2v2(self):
        indx = np.zeros((len(self.data),30,100))
        for i in range(len(self.data)):
            indx[i] = self.data2v(i)
        return indx

    def __len__(self):
        return len(self._idx2token)
class LSTM_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        vec = self.data[idx][0]
        vec = self.transform(vec)
        label = self.data[idx][1]
        return vec, label
class LSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional, dropout):
        super(LSTM_Net, self).__init__()
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * (1+bidirectional), 1),
            nn.Sigmoid()
 )

    def forward(self, batch):
        output, (_, _) = self.rnn(batch)
        output = output.mean(1)
        logit = self.classifier(output)
        return logit
from sklearn.metrics import f1_score
def evaluation(output, labels):
    output[output >=0.5 ] = 1
    output[output < 0.5] = 0
    output = output.detach().numpy()
    score = f1_score(labels, output, average='micro')
    return score
trainX = pd.read_csv(train_path)
trainY = pd.read_csv(trainY_path)
w2v, ptt_remove_stop = read_data(trainX)
vocab = Vocab(w2v, ptt_remove_stop)
data = vocab.data2v2()
train_data = data[:10000]
valid_data = data[10000:]
train_data = list(zip(train_data, trainY['label'].values[:10000]))
valid_data = list(zip(valid_data, trainY['label'].values[10000:]))
transform = transforms.Compose([transforms.ToTensor()])
train_set = LSTM_dataset(train_data, transform)
valid_set = LSTM_dataset(valid_data, transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True)

model = LSTM_Net(100,256,2,True,0.5)
model = model.double()

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
n1 = len(train_loader)
n2 = len(valid_loader)
num_epoch = 10
for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    train_f1 = 0
    for idx, (seq, label) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = seq.shape[0]
        output = model(seq.view(batch_size,30,100))
        label = label.double()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        f1 = evaluation(output, label)
        train_f1 += f1
        train_loss += loss.item()
    print(f'\nTrain [Epoch:{epoch+1}/{num_epoch}],  loss : {train_loss},  f1 : {train_f1/n1}')
            
    model.eval()
    with torch.no_grad():
        train_loss, train_f1= 0,0
        for idx, (seq, label) in enumerate(valid_loader):
            batch_size = seq.shape[0]
            output = model(seq.view(batch_size,30,100))
            label = label.double()
            loss = loss_fn(output, label)
            f1 = evaluation(output, label)                
            train_f1 += f1
            train_loss += loss.item()
    print(f'\nValid  [Epoch:{epoch+1}/{num_epoch}],  loss : {train_loss},  f1 : {train_f1/n2}') 
    model.train()