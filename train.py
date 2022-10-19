import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys,ast
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from itertools import count
import tensorflow as tf
from dnn_models import get_model
from dataloader import get_dataloader
from thop import profile

class PenaltyLoss(nn.modules.loss._Loss):
    def __init__(self,alpha = 1, win_len_init =8000):
        super(PenaltyLoss, self).__init__()
        self.max_len = 10000
        self.alpha = alpha
        self.w_init = win_len_init
        self.w_len_history = deque(maxlen=self.max_len)
        
    def forward(self, w_len, loss):
        loss =  loss.detach()
        if len(self.w_len_history)< self.max_len:
            self.w_len_history.append(w_len.detach())
            return 0
        w_len_avg = torch.stack(list(self.w_len_history),dim=0).mean()

        penalty = torch.clamp((w_len-w_len_avg)/w_len_avg,min = 0)
        penalty = self.alpha*penalty*loss
        self.w_len_history.append(w_len.detach())

        return penalty


def train(model,dataset,w_len,w_shift,mask,hard_mask,sampling,penalty,batch_size,lr,n_val,valid_every,seed,save,print_training,comment):
    if mask.lower()  =="false":
        mask=''
    if sampling.lower()  =="false":
        sampling=''
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs=16000
    wlen=int(fs*w_len/1000.00)
    wshift=int(fs*w_shift/1000.00)

    #setting up model
    
    CNN_net = get_model(model,wlen,fs,mask,hard_mask,sampling).to(device)

    #setting up dataloader
    data_loader_train,data_loader_val = get_dataloader(dataset,wlen,wshift)
    train_iter = iter(data_loader_train)
    # RMSprop
    optimizer_CNN = optim.Adam([
                {'params': CNN_net.CNN.conv.parameters()},
                {'params': CNN_net.CNN.bn.parameters()},
                {'params': CNN_net.CNN.ln.parameters()},
                {'params': CNN_net.CNN.ln0.parameters()},
                {'params': CNN_net.DNN1_net.parameters()},
                {'params': CNN_net.DNN2_net.parameters()}],  lr=lr) 
    optimizer_window = optim.Adam([
                {'params': CNN_net.CNN.mask.parameters()},
                {'params': CNN_net.CNN.sampling.parameters()}],  lr=lr) 
    # optimizer = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
    cost = nn.NLLLoss()
    if penalty>0:
        if mask!='':
            penalty_window = PenaltyLoss(penalty, win_len_init = wlen)
        if sampling!='':
            penalty_sr = PenaltyLoss(penalty, win_len_init =fs//2)

    # setup job name
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    time_benchmark = datetime.now()
    job_name = f"{model}_{dataset}_{w_len}_{w_shift}_{mask}_{hard_mask}_{sampling}_{penalty}_{seed}"

    # setup checkpoint and log dirs
    checkpoints_path = Path('experiment') / "checkpoints" / job_name / start_time
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    writer = Logger(str(Path('experiment') / "logs" / job_name/ start_time))
    print(f"{job_name} -- {start_time}")
    # record training infos
    if print_training:
        pbar = tqdm(total=valid_every, ncols=0, desc="Train")
    running_train_loss, running_train_err, val_loss,val_err = deque(maxlen=100), deque(maxlen=100), deque(maxlen=len(data_loader_val)), deque(maxlen=len(data_loader_val))
    
    best_err = np.inf
    count_stop = 0
    
    # start training
    for step in count(start=1):

        try:
            batch,label,_ = next(train_iter)
        except:
            train_iter = iter(data_loader_train)
            batch,label,_ = next(train_iter)
        batch,label = batch.to(device),label.flatten().to(device)
        pout = CNN_net(batch)
        pred=torch.max(pout,dim=1)[1]
        loss = cost(pout, label.long())
        optimizer_CNN.zero_grad()
        optimizer_window.zero_grad()
        # optimizer.zero_grad()
        if penalty>0:
            sum_loss = loss
            if mask!='':
                sum_loss = sum_loss + penalty_window(CNN_net.get_window().sum(),loss)
            if sampling!='':
                sum_loss = sum_loss + penalty_sr(CNN_net.get_sr().sum(),loss)
            sum_loss.backward()

        else:
            loss.backward()
        optimizer_CNN.step()
        optimizer_CNN.zero_grad()
        # optimizer.step()
        if step % 32==1: #set to 1 to avoid updating on validation epoch
            optimizer_window.step()
            optimizer_window.zero_grad()

        CNN_net.bounding() # thresholding masking value

        err = torch.mean((pred!=label.long()).float())
        running_train_loss.append(loss.item())
        running_train_err.append(err.item())

        avg_train_loss = sum(running_train_loss) / len(running_train_loss)
        avg_train_err = sum(running_train_err) / len(running_train_err)

        if print_training and step % (valid_every//10) == 0:
            pbar.update(valid_every//10)
            pbar.set_postfix(loss=avg_train_loss, err=avg_train_err, wlen = CNN_net.get_window().item(),sr = CNN_net.get_sr().item())
        
        if step % valid_every == 0:
            snt_err = []
            sentence_idx = -1
            pout_sentence = []
            if print_training:
                pbar.reset()
            CNN_net.eval()
            val_iter = iter(data_loader_val)
            for val_batch in range(len(data_loader_val)):
                batch,label,snt_id = next(val_iter)
                batch,label,snt_id = batch.to(device),label.flatten().to(device),snt_id.flatten().to(device)

                with torch.no_grad():
                    pout = CNN_net(batch)
                    pred=torch.max(pout,dim=1)[1]
                    loss = cost(pout, label.long())
                    err = torch.mean((pred!=label.long()).float())
                    val_loss.append(loss.item())
                    val_err.append(err.item())
                    #compute sentence error
                    for i,sample in enumerate(snt_id):
                        if sentence_idx==-1:
                            sentence_idx=sample
                        if sentence_idx!=sample:
                            pout_sentence = torch.stack(pout_sentence,dim=0)
                            [val,best_class]=torch.max(torch.sum(pout_sentence,dim=0),0)
                            snt_err.append(best_class.item()!=snt_lab.item())
                            pout_sentence = []
                            sentence_idx=sample
                        pout_sentence.append(pout[i])
                        snt_lab = label[i]
                        
            #for last iteration
            pout_sentence = torch.stack(pout_sentence,dim=0)
            [val,best_class]=torch.max(torch.sum(pout_sentence,dim=0),0)
            snt_err.append(best_class.item()!=snt_lab.item())

            #logging
            avg_valid_loss = sum(val_loss) / len(val_loss)
            avg_valid_err = sum(val_err) / len(val_err)
            avg_valid_err_snt = sum(snt_err) / len(snt_err)

            tqdm.write(f"Valid: {count_stop}, loss={avg_valid_loss:.3f}, err={avg_valid_err:.3f}, err_snt={avg_valid_err_snt:.3f}, tr_loss={avg_train_loss:.3f}, tr_err={avg_train_err:.3f}, length={CNN_net.get_window().item():.1f}, sr={CNN_net.get_sr().item():.1f}")
            writer.scalar_summary("Loss/train", avg_train_loss, step)
            writer.scalar_summary("Loss/val", avg_valid_loss, step)
            writer.scalar_summary("err/train", avg_train_err, step)
            writer.scalar_summary("err/val", avg_valid_err, step)
            writer.scalar_summary("err_snt/val", avg_valid_err_snt, step)
            writer.scalar_summary("mask_window",CNN_net.get_window().item(),step)
            writer.scalar_summary("sr",CNN_net.get_sr().item(),step)

            CNN_net.train()
            count_stop+=1
            #save                    
            if save and avg_valid_err_snt< best_err:
                best_err = avg_valid_err_snt
                torch.save(CNN_net, checkpoints_path / "model_raw.pkl") 

            if n_val==count_stop:
                print("Average time per step: {}".format((datetime.now()-time_benchmark).total_seconds()/step))
                macs, params = profile(CNN_net, inputs=(batch[0:1],))
                print("MACS: {}, Params: {}".format(macs, params))
                break



class Logger(object):
    """Tensorboard logger."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)
    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--model", type=str, default='SincNet')
    PARSER.add_argument("--dataset", type=str, default='TIMIT')
    PARSER.add_argument("--w_len", type=int, default=500)
    PARSER.add_argument("--w_shift", type=int, default=10)
    PARSER.add_argument("--mask", type=str, default='hann')
    PARSER.add_argument("--hard_mask", type=ast.literal_eval, default=True)
    PARSER.add_argument("--sampling", type=str, default='FFT')
    PARSER.add_argument("--penalty", type=float, default=0.1)
    PARSER.add_argument("--batch_size", type=int, default=128)
    PARSER.add_argument("--lr", type=float, default=0.001)
    PARSER.add_argument("--n_val", type=int, default=120) 
    PARSER.add_argument("--valid_every", type=int, default=800*4) #one epoch is 800 iteration, 4*50=200 epochs
    PARSER.add_argument("--seed", type=int, default=42)
    PARSER.add_argument("--save", type=ast.literal_eval, default=False)
    PARSER.add_argument("--print_training", type=ast.literal_eval, default=False)
    PARSER.add_argument("--comment", type=str, default='')
    train(**vars(PARSER.parse_args()))
