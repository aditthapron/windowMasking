## MASKING KERNEL FOR LEARNING ENERGY-EFFICIENT SPEECH REPRESENTATION

This repository contains codes used to reproduce speaker recognition results in "Masking Kernel for Learning Energy-Efficient Speech Representation" paper.
Masking kernel is introduced to optimize winow length and sampling rate of input speech as energy-efficient parameters, together with other parameters in the DNN model.
The method is compatible with most deep learning models, but we only include the [Sincnet](https://github.com/mravanelli/SincNet), a speaker recognition model using TIMIT corpus, and CNN in this repository.


## Instruction

### Dependencies
- pyTorch 1.10
- pysoundfile
- Scipy
- Numpy

### How to run
1. Generate normalized TIMIT files following instruction on [Sincnet](https://github.com/mravanelli/SincNet).

2. Run speaker recognition experiments using following code:

         
         python train.py --model SincNet --mask hamming --hard_mask True --sampling FFT --penalty 0.1
         

   The supported models are *SincNet* and *CNNNet* with the choise of masking filter, e.g. *gaussian*, *hamming* and *hann*.

   For more parameter configurations (such as training rate, training epoch, etc.), please check parser function in train.py. 
