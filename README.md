## Masking Kernel for Learning Energy-Efficient Representations for Speaker Recognition and Mobile Health

This repository contains codes used to reproduce speaker recognition results in "Masking Kernel for Learning Energy-Efficient Representations for Speaker Recognition and Mobile Health" paper.
Masking kernel is introduced to optimize winow length and sampling rate of input speech as energy-efficient parameters, together with other parameters in the DNN model.

To demonstate compatibility of our methods with various speech features and DNN model, we include [AM-MobileNet1D](https://github.com/joaoantoniocn/AM-MobileNet1D) branch for MFCC, Spectrogram and Raw audio, and [Sincnet](https://github.com/mravanelli/SincNet) on raw audio. For  SincNet please change branch to SincNet.
  


## Instruction

### Dependencies
- pyTorch 1.10
- pysoundfile
- Scipy
- Numpy
- [THOP](https://github.com/Lyken17/pytorch-OpCounter)  

### How to run
1. Generate normalized TIMIT files following instruction on [Sincnet](https://github.com/mravanelli/SincNet).

2. Edit 'PATH_TO_NORMALIZED_TIMIT_DATASET' in dataloader.py line 12 to where you keep TIMIT dataset. 

3. Run speaker recognition experiments using following code:

         
         python train.py --model MobileNetV2 --mask hamming --hard_mask True --sampling FFT --penalty 0.1
         

   The supported models are *MobileNetV2*, *MobileNetV2_MFCC* and *MobileNetV2_spectrogram* with the choise of masking filter, e.g. *gaussian*, *hamming* and *hann*.

   For more parameter configurations (such as training rate, training epoch, etc.), please check parser function in train.py. 
   A sample of training result is provided in Results folder of SincNet branch. 


