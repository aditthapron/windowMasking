import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import soundfile as sf
import sys
import os

class TIMIT(Dataset):
    """ Data loader"""
    def __init__(self, lst:list, wlen:int, wshift:int, fact_amp=0):
        self.data_folder = '/home/aditthapron/2022/Masking/TIMIT/'
        self.wav_lst = ReadList(lst)
        self.lab_dict=np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()
        self.N_snt=len(self.wav_lst)
        self.wlen = wlen
        self.fact_amp = fact_amp
        self.wshift = wshift
        self.fr = np.zeros(self.N_snt) #self.idx keep number of accumalating sample for each wav
        self.total_fr = 0
        #get total number of sample

        for i in range(self.N_snt):
            f = sf.SoundFile(self.data_folder+self.wav_lst[i])
            self.total_fr += int((f.frames-self.wlen)/(self.wshift))
            self.fr[i] = self.total_fr

    def __getitem__(self, idx):

        rand_amp = np.random.uniform(1.0-self.fact_amp,1+self.fact_amp)
        #get which file to read
        for i in range(self.N_snt):
            if idx < self.fr[i]:
                [signal, fs] = sf.read(self.data_folder+self.wav_lst[i])
                channels = len(signal.shape)
                if channels == 2:
                    signal = signal[:,0]
                if i>0:
                    idx=idx-self.fr[i-1]
                begin = int(idx*self.wshift)
                end = begin + self.wlen

                signal = signal[begin:end] * rand_amp
                lab = self.lab_dict[self.wav_lst[i].lower()]
                return torch.FloatTensor(signal),torch.FloatTensor([lab]),i

    def __len__(self):
        return self.total_fr

class TIMIT_sentence(Dataset):
    """ Data loader"""
    def __init__(self, lst:list, wlen:int, wshift:int, fact_amp=0):
        self.data_folder = '/home/aditthapron/2022/Masking/TIMIT/'
        self.wav_lst = ReadList(lst)
        self.lab_dict=np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()
        self.N_snt=len(self.wav_lst)
        self.wlen = wlen
        self.fact_amp = fact_amp

    def __getitem__(self, idx):
        rand_amp = np.random.uniform(1.0-self.fact_amp,1+self.fact_amp)
        [signal, fs] = sf.read(self.data_folder+self.wav_lst[idx])
        channels = len(signal.shape)
        if channels == 2:
            signal = signal[:,0]
        signal = signal * rand_amp
        lab = self.lab_dict[self.wav_lst[i].lower()]
        return torch.FloatTensor(signal),torch.FloatTensor([lab])

    def __len__(self):
        return self.N_snt

def str_to_class(name):
    return getattr(sys.modules[__name__],name)

def ReadList(list_file):
    f=open(list_file,"r")
    lines=f.readlines()
    list_sig=[]
    for x in lines:
        list_sig.append(x.rstrip().upper())
    f.close()
    return list_sig



def get_dataloader(loader_name,wlen,wshift):
    dataset_train = str_to_class(loader_name)('data_lists/TIMIT_train.scp',wlen,wshift,fact_amp=0.2)
    dataset_val = str_to_class(loader_name)('data_lists/TIMIT_test.scp',wlen,wshift,fact_amp=0)
    data_loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=len(os.sched_getaffinity(0)),drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=len(os.sched_getaffinity(0)),drop_last=False)
    return data_loader_train,data_loader_val


if __name__ == '__main__':
    dataset_train = TIMIT('data_lists/TIMIT_train.scp',int(16000*0.2),int(16000*0.01))
    data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=2,drop_last=True)
    train_iter = iter(data_loader_train)
    for i in range(1):
        out = next(train_iter)

