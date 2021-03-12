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

test_path = sys.argv[1]
ans_path = sys.argv[2]


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
    return ptt_remove_stop    

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
nlp = spacy.load('en_core_web_md')
tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)    
testX = pd.read_csv(test_path)
ptt_remove_stop_test = read_data(testX)    
w2v = np.load('w2v.npy',allow_pickle=True)
ww = []
for i in range(len(w2v)):
    ww.append((w2v[i][0],w2v[i][1]))
w2v = ww    
vocab_test = Vocab(w2v, ptt_remove_stop_test)
transform = transforms.Compose([transforms.ToTensor()])
test_data = vocab_test.data2v2()   
test_label = np.full(test_data.shape[0],1)
test_data = list(zip(test_data, test_label))
test_set = LSTM_dataset(test_data, transform)
test_loader = DataLoader(test_set, batch_size=32)

use_gpu = torch.cuda.is_available()
model = LSTM_Net(100,256,2,True,0.5)
model = model.double()
model.load_state_dict(torch.load('model123.pkl'))

if use_gpu:
    model.cuda()
model.eval()

predict_value = []
for seq, _ in test_loader:
    if use_gpu:
        seq = seq.cuda()
    batch_size = seq.shape[0]
    output = model(seq.view(batch_size,30,100))
    predict_value += output.tolist()
    
for idx,i in enumerate(predict_value):
    if i[0] >= 0.5:
        predict_value[idx] = 1
    else:
        predict_value[idx] = 0
sub = pd.DataFrame()
sub['id'] = np.arange(len(predict_value))
sub['label'] = predict_value
sub.to_csv(ans_path, index = False)