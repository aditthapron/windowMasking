## MASKING KERNEL FOR LEARNING ENERGY-EFFICIENT SPEECH REPRESENTATION

This repository contains codes used to reproduce speaker recognition results in "MASKING KERNEL FOR LEARNING ENERGY-EFFICIENT SPEECH REPRESENTATION" paper.
The speaker recognition model was extended from [Sincnet](https://github.com/mravanelli/SincNet) using TIMIT corpus.
Masking kernel optimizes winow length and sampling rate of input speech as energy-efficient parameters, together with other parameters in the SincNet model.

## Instruction

### Dependencies
- pyTorch 1.10
- pysoundfile
- Scipy
- Numpy

### How to run
** 1. Normalize TIMIT data following instruction on [Sincnet](https://github.com/mravanelli/SincNet).
** 2. Run speaker recognition experiments using following code:
``
python train.py --model SincNet --mask hamming --hard_mask True --sampling FFT --penalty 0.1
``
The supported models are *SincNet* and *CNNNet* with the choise of masking filter, e.g. *gaussian* *hamming* *hann*.
For more config of parameters (such as training rate, training epoch, etc.), please check parser in train.py. 