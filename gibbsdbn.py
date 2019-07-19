import numpy as np
from traindbn import *
import time

with np.load('test.npz') as data:
    conv0 = data['conv0']

with open('lpc.txt', 'r') as dat:
    corpus = munge(dat, min_len=9)


seqs = one_hot_seqs(len(atoi), corpus)

for gibbs_start in seqs:

    for cur in clamped_conv_rbm_gibbs(gibbs_start, conv0):
        print(''.join(itoa[c] for c in cur))
    time.sleep(10)
