## Masking Kernel for Learning Energy-Efficient Representations for Speaker Recognition and Mobile Health

SincNet model

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

         
         python train.py --model SincNet --mask hamming --hard_mask True --sampling FFT --penalty 0.1
         

   The supported models are *SincNet* and *CNNNet* with the choise of masking filter, e.g. *gaussian*, *hamming* and *hann*.

   For more parameter configurations (such as training rate, training epoch, etc.), please check parser function in train.py. 
   A sample of training result is provided in Results folder. 

### Overfitting to penalty term
Training and validatio losses (tr_err,te_err) reported in the results do not include penalty term.
Based on the result of hamming window with a random seed value of 1, training errors (tr_err) are in a range of (0.005,0.020) until epoch 88 that training loss keeps increasing, which is caused by penalty loss value. At around epoch 88, the optimizer overoptimizes the penalty term -- keep decreasing the signal length with less optimization done on the main network -- and, in turn, increases training loss. Hence, we only include results up to epoch 100 in the paper and report best performance before the overoptimization occurs (based on training loss).

