# Created by
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018
# ------------------
# Modified by
# Joao Antonio Chagas Nunes
# Universidade Federal de Pernambuco

# Add AM-Softmax loss function
# December 2018
# ------------------
# Modified by
# Joao Antonio Chagas Nunes
# Universidade Federal de Pernambuco

# Add MobileNet1D and AM-MobileNet1D
# January 2020
# ------------------

# Jan 2023
# Modified by double-blind 
# Add Masking layers

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchaudio import transforms 
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



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self,wlen,fs,mask,hard_mask,sampling, num_classes=462, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.input_dim = wlen
        self.fs=fs

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(1, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            LinearTruncate(self.last_channel, num_classes),
            nn.LogSoftmax(dim=1),
        )

        self.normalize = nn.BatchNorm1d(fs//2)
        self.maxpool1d = nn.MaxPool1d(3, stride=2)

        #masking
        self.mask = mask
        self.sampling = sampling
        if self.mask is not '':
            self.mask = Masking(self.input_dim,mask,hard_mask)

        if self.sampling is not '':
            if sampling == 'Sinc':
                self.sampling = SamplingRateSinc(self.input_dim)
            if sampling == 'FFT':
                self.sampling = SamplingRateFFT(self.input_dim)


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if(len(x.shape) ==2):
            # x = self.normalize(x)
            x = x.reshape([x.shape[0], 1, x.shape[1]])
            # x = self.maxpool1d(x)
        #Maskingd

        if self.mask is not '':
            x = self.mask(x)
        if self.sampling is not '':
            x = self.sampling(x)

        x = self.features(x)
        x = x.mean(2)
        x = self.classifier(x)
        return x
    def get_window(self):
        if self.mask:
            return self.mask.get_window()
        else:
            return torch.tensor(self.input_dim)
    def bounding(self):
        if self.mask:
            self.mask.bounding()

    def get_sr(self):
        if self.sampling:
            return self.sampling.get_sr()
        else:
            return torch.tensor(self.fs//2)

class MobileNetV2_MFCC(nn.Module):
    def __init__(self,wlen,fs,mask,hard_mask,sampling, num_classes=462, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2_MFCC, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.input_dim = wlen
        self.fs=fs

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(13, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            LinearTruncate(self.last_channel, num_classes),
            nn.LogSoftmax(dim=1),
        )

        self.normalize = nn.BatchNorm1d(fs//2)
        self.maxpool1d = nn.MaxPool1d(3, stride=2)

        #masking
        self.mask = mask
        self.sampling = sampling
        if self.mask is not '':
            self.mask = Masking(self.input_dim,mask,hard_mask)

        if self.sampling is not '':
            if sampling == 'Sinc':
                self.sampling = SamplingRateSinc(self.input_dim)
            if sampling == 'FFT':
                self.sampling = SamplingRateFFT(self.input_dim)

        #MFCC
        self.MFCC_transform = transforms.MFCC(sample_rate=fs,n_mfcc=13, melkwargs={"n_fft": 1024, "hop_length": 128, "n_mels": 40, "center": False})

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if(len(x.shape) ==2):
            # x = self.normalize(x)
            x = x.reshape([x.shape[0], 1, x.shape[1]])
            # x = self.maxpool1d(x)

        #Masking
        if self.mask is not '':
            x = self.mask(x)
        if self.sampling is not '':
            x = self.sampling(x)


        x = self.MFCC_transform(x)
        x = x.reshape([x.shape[0], x.shape[2], x.shape[3]])

        x = self.features(x)
        x = x.mean(2)
        x = self.classifier(x)
        return x
    def get_window(self):
        if self.mask:
            return self.mask.get_window()
        else:
            return torch.tensor(self.input_dim)
    def bounding(self):
        if self.mask:
            self.mask.bounding()

    def get_sr(self):
        if self.sampling:
            return self.sampling.get_sr()
        else:
            return torch.tensor(self.fs//2)

class MobileNetV2_spectrogram(nn.Module):
    def __init__(self,wlen,fs,mask,hard_mask,sampling, num_classes=462, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2_spectrogram, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.input_dim = wlen
        self.fs=fs

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(40, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            LinearTruncate(self.last_channel, num_classes),
            nn.LogSoftmax(dim=1),
        )

        self.normalize = nn.BatchNorm1d(fs//2)
        self.maxpool1d = nn.MaxPool1d(3, stride=2)

        #masking
        self.mask = mask
        self.sampling = sampling
        if self.mask is not '':
            self.mask = Masking(self.input_dim,mask,hard_mask)

        if self.sampling is not '':
            if sampling == 'Sinc':
                self.sampling = SamplingRateSinc(self.input_dim)
            if sampling == 'FFT':
                self.sampling = SamplingRateFFT(self.input_dim)

        #Spectrogram
        self.spectrogram_transform = transforms.MelSpectrogram(sample_rate=fs,n_fft= 1024, hop_length= 128, n_mels= 40, center= False)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if(len(x.shape) ==2):
            # x = self.normalize(x)
            x = x.reshape([x.shape[0], 1, x.shape[1]])
            # x = self.maxpool1d(x)

        #Masking
        if self.mask is not '':
            x = self.mask(x)
        if self.sampling is not '':
            x = self.sampling(x)


        x = self.spectrogram_transform(x)
        x = x.reshape([x.shape[0], x.shape[2], x.shape[3]])

        x = self.features(x)
        x = x.mean(2)
        x = self.classifier(x)
        return x
    def get_window(self):
        if self.mask:
            return self.mask.get_window()
        else:
            return torch.tensor(self.input_dim)
    def bounding(self):
        if self.mask:
            self.mask.bounding()

    def get_sr(self):
        if self.sampling:
            return self.sampling.get_sr()
        else:
            return torch.tensor(self.fs//2)
            
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
                   

def str_to_class(name):
    return getattr(sys.modules[__name__], name)
def get_model(model_name,wlen,fs,mask,hard_mask,sampling):
    return str_to_class(model_name)(wlen,fs,mask,hard_mask,sampling)


if __name__ == '__main__':
    x = torch.zeros(2,3200)
    model = MobileNetV2_MFCC('hann',True,'FFT',3200)
    out = model(x)
