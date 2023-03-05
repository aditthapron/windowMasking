# Created by
# Mirco Ravanelli 
# Mila - University of Montreal 

# Jan 2023
# Modified by double-blind 
# Add Masking layers

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
from scipy.fft import rfftfreq

class Masking(nn.Module):
    def __init__(self,dim,mask,hard):
        super(Masking,self).__init__()
        self.mask = mask
        self.hard_mask = hard
        self.input_dim = dim
        self.epsilon = 1e-12

        if self.mask == 'gaussian':
            self.w_space = nn.Parameter(torch.linspace(-self.input_dim//2+1,self.input_dim//2+1,self.input_dim,dtype=torch.float, requires_grad=False))
            self.factor = float(self.input_dim//4/(math.sqrt(-math.log(self.epsilon)/0.5)))
            self.r_masking = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        else:
            self.w_space = nn.Parameter(torch.linspace(0,self.input_dim,self.input_dim,dtype=torch.float, requires_grad=False))
            self.factor = float(self.input_dim//2)
            self.r_masking = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self,x):
        if self.mask == 'gaussian':
            mask = torch.exp(-0.5*(self.w_space/(self.factor*self.r_masking))** 2)
            r_mask = int(math.sqrt(-math.log(self.epsilon)/0.5) *  self.r_masking*self.factor*2)
            mask = mask[(self.input_dim-r_mask)//2:(self.input_dim+r_mask)//2]

        elif self.mask == 'hamming':
            r_mask = int(self.r_masking*self.factor)
            mask = 0.54-0.46*torch.cos((2*torch.Tensor([math.pi]).to(self.w_space.device)*self.w_space)/((self.r_masking*self.factor+ self.epsilon)-1))
            mask = mask[:r_mask]

        elif self.mask == 'hann':
            r_mask = int(self.r_masking*self.factor)
            mask = 0.5-0.5*torch.cos((2*torch.Tensor([math.pi]).to(self.w_space.device)*self.w_space)/((self.r_masking*self.factor+ self.epsilon)-1))
            mask = mask[:r_mask]
        elif self.mask == 'tukey_5':
            r_mask = int(self.r_masking*self.factor)
            r=0.5
            mask = torch.where(self.w_space<=r*r_mask/2, 0.5+0.5*torch.cos((2*torch.Tensor([math.pi]).to(self.w_space.device)*(self.w_space-r*self.r_masking*self.factor/2))/((r*self.r_masking*self.factor+ self.epsilon)-1)),
                torch.where(self.w_space<(1-r/2)*r_mask,torch.ones_like(self.w_space),0.5+0.5*torch.cos((2*torch.Tensor([math.pi]).to(self.w_space.device)*(self.w_space-1+r*self.r_masking*self.factor/2))/((r*self.r_masking*self.factor+ self.epsilon)-1))))
            mask = mask[:r_mask]
        elif self.mask == 'tukey_1':
            r_mask = int(self.r_masking*self.factor)
            r=0.1
            mask = torch.where(self.w_space<=r*r_mask/2, 0.5+0.5*torch.cos((2*torch.Tensor([math.pi]).to(self.w_space.device)*(self.w_space-r*self.r_masking*self.factor/2))/((r*self.r_masking*self.factor+ self.epsilon)-1)),
                torch.where(self.w_space<(1-r/2)*r_mask,torch.ones_like(self.w_space),0.5+0.5*torch.cos((2*torch.Tensor([math.pi]).to(self.w_space.device)*(self.w_space-1+r*self.r_masking*self.factor/2))/((r*self.r_masking*self.factor+ self.epsilon)-1))))
            mask = mask[:r_mask]
        elif self.mask == 'rectangle':
            r_mask = self.r_masking*self.factor
            mask = torch.where(torch.logical_and(self.w_space>(self.input_dim-int(r_mask))//2,self.w_space<(self.input_dim+int(r_mask))//2),(r_mask+ self.epsilon)/(r_mask+ self.epsilon),0/(r_mask+ self.epsilon))
            mask = mask[:r_mask]
        else:
            raise NameError('Wrong mask name')

        if self.hard_mask:
            mask = torch.where(mask>self.epsilon,(mask+ self.epsilon)/(mask+ self.epsilon),0/(mask+ self.epsilon))
            # mask = STEFunction.apply(mask)

        x = x[:,:,(self.input_dim-r_mask)//2:(self.input_dim+r_mask)//2]*mask

        return x

    def get_window(self):
        if self.mask == 'gaussian':
            return math.sqrt(-math.log(self.epsilon)/0.5) *  self.r_masking*self.factor*2
        else:
            return self.r_masking*self.factor

    def bounding(self):
        with torch.no_grad():
            self.r_masking[:] = self.r_masking.clamp(1e-2, 2) 

class SamplingRateFFT(nn.Module):
    def __init__(self,dim,init_freq=16000,smooth=100):
        super(SamplingRateFFT,self).__init__()
        self.init_freq = init_freq
        self.smooth = smooth
        self.dim=dim
        self.factor = float(8000)
        self.r_masking = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        

    def forward(self,x):
        self.w_space = nn.Parameter(torch.tensor(rfftfreq(x.shape[-1],1/self.init_freq)), requires_grad=True)
        f_signal = torch.fft.rfft(x)
        mask = torch.min(torch.ones_like(self.w_space),torch.max(torch.zeros_like(self.w_space),-(self.w_space - self.factor*self.r_masking)/self.smooth))
        f_signal = f_signal*mask
        f_signal = f_signal[:,:,:int(((self.factor*self.r_masking))/self.factor*self.w_space.shape[0])]

        return torch.fft.irfft(f_signal).float()

    def get_sr(self):
        return self.r_masking*self.factor

    def bounding(self):
        with torch.no_grad():
            self.r_masking[:] = self.r_masking.clamp(1e-2, 1-1e-2)

class SamplingRateSinc(nn.Module):
    def __init__(self,dim,sample_rate=16000):
        super(SamplingRateSinc,self).__init__()
        self.low_pass = SincConv_lowpass(251,sample_rate,padding=125)
        self.sample_rate = sample_rate

    def forward(self,x):
        return self.low_pass(x)

    def get_sr(self):
        return self.low_pass.band_hz_

    def bounding(self):
        with torch.no_grad():
            self.low_pass.band_hz_[:] = self.low_pass.band_hz_.clamp(20, self.sample_rate//2)

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.001).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)



class SincNet(nn.Module):
    def __init__(self,wlen,fs,mask=False,hard_mask=True,sampling=False):
        super(SincNet,self).__init__()
        self.mask = mask
        self.wlen = wlen
        self.fs = fs
        self.sampling = sampling
        CNN_option = {'input_dim': wlen,
                        'fs': fs,
                        'cnn_N_filt':[80,60,60],
                        'cnn_len_filt':[251,5,5],
                        'cnn_max_pool_len':[3,3,3],
                        'cnn_use_laynorm_inp':True,
                        'cnn_use_batchnorm_inp':False,
                        'cnn_use_laynorm':[True,True,True],
                        'cnn_use_batchnorm':[False,False,False],
                        'cnn_act':['leaky_relu','leaky_relu','leaky_relu'],
                        'cnn_drop':[0.0,0.0,0.0]}
        self.CNN = Sinc_CNN_CNN(CNN_option,mask,hard_mask,sampling)
        DNN_1_option = {'input_dim': self.CNN.out_dim,
                        'fc_lay':[2048,2048,2048],
                        'fc_drop':[0.0,0.0,0.0],
                        'fc_use_laynorm_inp':True,
                        'fc_use_batchnorm_inp':False,
                        'fc_use_batchnorm':[True,True,True],
                        'fc_use_laynorm':[False,False,False],
                        'fc_act':['leaky_relu','leaky_relu','leaky_relu']}
        DNN_2_option = {'input_dim':2048,
                        'fc_lay':[462],
                        'fc_drop':[0.0],
                        'fc_use_laynorm_inp':False,
                        'fc_use_batchnorm_inp':False,
                        'fc_use_batchnorm':[False],
                        'fc_use_laynorm':[False],
                        'fc_act':['softmax']}
        
        self.DNN1_net=MLP(DNN_1_option)
        self.DNN2_net=MLP(DNN_2_option)

    def forward(self,x):
        return self.DNN2_net(self.DNN1_net(self.CNN(x)))

    def get_window(self):
        if self.mask:
            return self.CNN.mask.get_window()
        else:
            return torch.tensor(self.wlen)
    def bounding(self):
        if self.mask:
            self.CNN.mask.bounding()

    def get_sr(self):
        if self.sampling:
            return self.CNN.sampling.get_sr()
        else:
            return torch.tensor(self.fs//2)
        # if self.mask:
        #     return self.CNN.mask.get_window()
        # else:
        #     return torch.tensor(self.wlen)

class CNNNet(nn.Module):
    def __init__(self,wlen,fs,mask=False,hard_mask=True,sampling=False):
        super(CNNNet,self).__init__()
        self.mask = mask
        self.wlen = wlen
        self.fs = fs
        self.sampling = sampling
        CNN_option = {'input_dim': wlen,
                        'fs': fs,
                        'cnn_N_filt':[80,60,60],
                        'cnn_len_filt':[251,5,5],
                        'cnn_max_pool_len':[3,3,3],
                        'cnn_use_laynorm_inp':True,
                        'cnn_use_batchnorm_inp':False,
                        'cnn_use_laynorm':[True,True,True],
                        'cnn_use_batchnorm':[False,False,False],
                        'cnn_act':['leaky_relu','leaky_relu','leaky_relu'],
                        'cnn_drop':[0.0,0.0,0.0]}
        self.CNN = CNN_CNN_CNN(CNN_option,mask,hard_mask,sampling)
        DNN_1_option = {'input_dim': self.CNN.out_dim,
                        'fc_lay':[2048,2048,2048],
                        'fc_drop':[0.0,0.0,0.0],
                        'fc_use_laynorm_inp':True,
                        'fc_use_batchnorm_inp':False,
                        'fc_use_batchnorm':[True,True,True],
                        'fc_use_laynorm':[False,False,False],
                        'fc_act':['leaky_relu','leaky_relu','leaky_relu']}
        DNN_2_option = {'input_dim':2048,
                        'fc_lay':[462],
                        'fc_drop':[0.0],
                        'fc_use_laynorm_inp':True,
                        'fc_use_batchnorm_inp':False,
                        'fc_use_batchnorm':[False],
                        'fc_use_laynorm':[False],
                        'fc_act':['softmax']}
        
        self.DNN1_net=MLP(DNN_1_option)
        self.DNN2_net=MLP(DNN_2_option)

    def forward(self,x):
        return self.DNN2_net(self.DNN1_net(self.CNN(x)))

    def get_window(self):
        if self.mask:
            return self.CNN.mask.get_window()
        else:
            return torch.tensor(self.wlen)
    def bounding(self):
        if self.mask:
            self.CNN.mask.bounding()
    def get_sr(self):
        if self.sampling:
            return self.CNN.sampling.get_sr()
        else:
            return torch.tensor(self.fs//2)
class SincConv_lowpass(nn.Module):

    """Sinc-based convolution
    adapted from SincConv_fast to perform only low-pass filter
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, kernel_size, sample_rate=16000, input_dim=6400, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_lowpass,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.kernel_size = kernel_size
        self.input_dim = input_dim
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate


        high_hz = self.sample_rate // 2
        self.band_hz_ = nn.Parameter(torch.Tensor([high_hz]))
        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        
        f_times_t_high = torch.matmul(self.band_hz_, self.n_)

        band_pass_left=(torch.sin(f_times_t_high)/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*self.band_hz_.view(1, -1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*self.band_hz_)
        

        self.filters = (band_pass).view(1,1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, input_dim=6400, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        low = self.min_low_hz  + torch.abs(self.low_hz_)
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 


class Sinc_CNN_CNN(nn.Module):
    
    def __init__(self,options, mask=None, hard_mask = False,sampling = None):
        super(Sinc_CNN_CNN,self).__init__()
        self.cnn_N_filt=options['cnn_N_filt']
        self.cnn_len_filt=options['cnn_len_filt']
        self.cnn_max_pool_len=options['cnn_max_pool_len']
        self.cnn_act=options['cnn_act']
        self.cnn_drop=options['cnn_drop']
        self.cnn_use_laynorm=options['cnn_use_laynorm']
        self.cnn_use_batchnorm=options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
        self.input_dim=int(options['input_dim'])
        self.fs=options['fs']
        self.mask = mask
        self.sampling = sampling
        #Masking
        if self.mask is not '':
            self.mask = Masking(self.input_dim,mask,hard_mask)

        if sampling is not '':
            if sampling == 'Sinc':
                self.sampling = SamplingRateSinc(self.input_dim)
            if sampling == 'FFT':
                self.sampling = SamplingRateFFT(self.input_dim)

        self.N_cnn_lay=len(options['cnn_N_filt'])
        self.conv  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
    
        if self.cnn_use_laynorm_inp:
            self.ln0=LayerNorm(self.input_dim)
           
        if self.cnn_use_batchnorm_inp:
            self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
       
        current_input=self.input_dim 
       
        for i in range(self.N_cnn_lay):
            N_filt=int(self.cnn_N_filt[i])
            len_filt=int(self.cnn_len_filt[i])
            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
            # activation
            self.act.append(act_fun(self.cnn_act[i]))
            # layer norm initialization         
            self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
            
            if i==0:
                self.conv.append(SincConv_fast(self.cnn_N_filt[0],self.cnn_len_filt[0],self.fs,current_input))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])
        self.out_dim=current_input*N_filt

    def forward(self, x):

        batch=x.shape[0]
        seq_len=x.shape[1]
        x=x.view(batch,1,seq_len)
        #Masking
        if self.mask is not '':
            x = self.mask(x)
        if self.sampling is not '':
            x = self.sampling(x)


        if bool(self.cnn_use_laynorm_inp):
            x=self.ln0((x))
        if bool(self.cnn_use_batchnorm_inp):
            x=self.bn0((x))

       

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                if i==0:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
                else:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
        x = torch.swapaxes(x,1,2) 
        x = x.reshape(batch,-1)
        return x

class CNN_CNN_CNN(nn.Module):
    
    def __init__(self,options, mask=None, hard_mask = False,sampling = None):
        super(CNN_CNN_CNN,self).__init__()
        self.cnn_N_filt=options['cnn_N_filt']
        self.cnn_len_filt=options['cnn_len_filt']
        self.cnn_max_pool_len=options['cnn_max_pool_len']
        self.cnn_act=options['cnn_act']
        self.cnn_drop=options['cnn_drop']
        self.cnn_use_laynorm=options['cnn_use_laynorm']
        self.cnn_use_batchnorm=options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
        self.input_dim=int(options['input_dim'])
        self.fs=options['fs']
        self.mask = mask
        self.sampling = sampling
        #Masking
        if self.mask is not '':
            self.mask = Masking(self.input_dim,mask,hard_mask)

        if self.sampling is not '':
            if sampling == 'Sinc':
                self.sampling = SamplingRateSinc(self.input_dim)
            if sampling == 'FFT':
                self.sampling = SamplingRateFFT(self.input_dim)
       
        self.N_cnn_lay=len(options['cnn_N_filt'])
        self.conv  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
    
        if self.cnn_use_laynorm_inp:
            self.ln0=LayerNorm(self.input_dim)
           
        if self.cnn_use_batchnorm_inp:
            self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
       
        current_input=self.input_dim 
       
        for i in range(self.N_cnn_lay):
            N_filt=int(self.cnn_N_filt[i])
            len_filt=int(self.cnn_len_filt[i])
            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
            # activation
            self.act.append(act_fun(self.cnn_act[i]))
            # layer norm initialization         
            self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
            
            if i==0:
                self.conv.append(nn.Conv1d(1, self.cnn_N_filt[i], self.cnn_len_filt[i]))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])
        self.out_dim=current_input*N_filt

    def forward(self, x):
        batch=x.shape[0]
        seq_len=x.shape[1]
        if bool(self.cnn_use_laynorm_inp):
            x=self.ln0((x))
        if bool(self.cnn_use_batchnorm_inp):
            x=self.bn0((x))

        x=x.view(batch,1,seq_len)

        #Masking
        if self.mask is not '':
            x = self.mask(x)
        if self.sampling is not '':
            x = self.sampling(x)

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                if i==0:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
                else:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        x = torch.swapaxes(x,1,2) # format to (batch,temporal,feature)
        x = x.reshape(batch,-1)
        return x
   

class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()
        
        self.input_dim=int(options['input_dim'])
        self.fc_lay=options['fc_lay']
        self.fc_drop=options['fc_drop']
        self.fc_use_batchnorm=options['fc_use_batchnorm']
        self.fc_use_laynorm=options['fc_use_laynorm']
        self.fc_use_laynorm_inp=options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp=options['fc_use_batchnorm_inp']
        self.fc_act=options['fc_act']
        self.wx  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        # input layer normalization
        if self.fc_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
        # input batch normalization    
        if self.fc_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
        self.N_fc_lay=len(self.fc_lay)
        current_input=self.input_dim
        # Initialization of hidden layers
        for i in range(self.N_fc_lay):
            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))
            # activation
            self.act.append(act_fun(self.fc_act[i]))
            add_bias=True
            # layer norm initialization
            self.ln.append(LayerNorm(self.fc_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
        
            if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
                add_bias=False
         
            # Linear operations
            self.wx.append(LinearTruncate(current_input, self.fc_lay[i],bias=add_bias))
            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
         
            current_input=self.fc_lay[i]
         
         
    def forward(self, x):
        # Applying Layer/Batch Norm
        if bool(self.fc_use_laynorm_inp):
            x=self.ln0((x))
        if bool(self.fc_use_batchnorm_inp):
            x=self.bn0((x))
        for i in range(self.N_fc_lay):
            
            if self.fc_act[i]!='linear':
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
                if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
                    x = self.drop[i](self.act[i](self.wx[i](x)))
            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))
                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))
                if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
                    x = self.drop[i](self.wx[i](x)) 
        return x

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def shrink_dim(x2,x1):
    #shrink x2 to have dimension as x1
    if x2.shape[-1]>x1.shape[-1]:
        x2 = x2[...,(x2.shape[-1]-x1.shape[-1])//2:-(x2.shape[-1]-x1.shape[-1])//2]
    return x2

def shrink_weight(x2,x1):
    #shrink x2 to have dimension 0 as dim in x1
    if x2.shape[-1]>x1.shape[-1]:
        x2 = x2[...,(x2.shape[-1]-x1.shape[-1])//2:-(x2.shape[-1]-x1.shape[-1])//2]
    return x2

class LinearTruncate(nn.Linear):
    def forward(self, input):
        return F.linear(input, shrink_weight(self.weight,input), self.bias)

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return shrink_dim(self.gamma,x) * (x - mean) / (std + self.eps) + shrink_dim(self.beta,x)

def act_fun(act_type):

    if act_type=="relu":
        return nn.ReLU()
            
    if act_type=="tanh":
        return nn.Tanh()
            
    if act_type=="sigmoid":
        return nn.Sigmoid()
           
    if act_type=="leaky_relu":
        return nn.LeakyReLU(0.2)
            
    if act_type=="elu":
        return nn.ELU()
                     
    if act_type=="softmax":
        return nn.LogSoftmax(dim=1)
        
    if act_type=="linear":
        return nn.LeakyReLU(1) # initializzed like this, but not used in forward!
            



def str_to_class(name):
    return getattr(sys.modules[__name__], name)
def get_model(model_name,wlen,fs,mask,hard_mask,sampling):
    return str_to_class(model_name)(wlen,fs,mask,hard_mask,sampling)

if __name__ == '__main__':
    x = torch.zeros(2,3200)
    model = SincNet(3200,16000,mask=False,hard_mask=True)
    out = model(x)
