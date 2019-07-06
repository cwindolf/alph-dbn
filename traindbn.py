import numpy as np
import string, itertools
from machines import AlphDBN, CategoricalFullHeightConvRBM as cRBM

np.seterr(over='raise')


ALPHABET = string.ascii_lowercase + '., '
N = len(ALPHABET)
itoa = dict(enumerate(ALPHABET))
atoi = dict((a, i) for i, a in enumerate(ALPHABET))

DEM = 'demons.txt'
LPC = 'lpc.txt'

# *************************************************************************** #
# Data processing

def munge(aloloc, min_len=7):
    aloloi = []
    for aloc in aloloc:
        aloi = []
        for c in aloc.strip().lower():
            if c in atoi:
                aloi.append(atoi[c])
        if len(aloi) >= min_len:
            aloloi.append(np.array(aloi, dtype=int))
    return aloloi


def test(corpus, min_len=30):
    for seq in corpus:
        if len(seq) < min_len: continue
        print('New Seq: ------------------')
        print(''.join(itoa[i] for i in seq))
        try:
            beta = float(input('Choose beta: '))
            beta_rate = float(input('Choose beta rate: '))
            max_beta = float(input('Choose max beta: '))
        except:
            print('OK bai.')
            return
        for res in dbn.gibbs(seq, 64, beta=beta, beta_rate=beta_rate, max_beta=max_beta):
            print(''.join(itoa[i] for i in res.argmax(axis=1)))


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # from tqdm import tqdm
    # import argparse

    # ap = argparse.ArgumentParser()
    # args = ap.parse_args()

    # *********************************************************************** #
    # Munge

    with open(LPC, 'r') as dat:
        lpc_corpus = munge(dat, min_len=15)

    with open(DEM, 'r') as dat:
        dem_corpus = munge(dat, min_len=45)

    # *********************************************************************** #
    # Train

    dbn = AlphDBN('lpcdbn', N)

    dbn.add_layer(cRBM('layer0', N, 3, 128, {
            'type': 0,
            'lambda': 0.05,
            'target': 0.2,
            'eta': 0.5,
        },  w_init=0.25, hb_init=-0.8, lr=2e-4, lrd=1e-5, wd=1.25,
        mo=0.5, categorical_inputs=True, itoa=itoa,
        pad_i=atoi[' ']))

    # dbn.load('.')
    dbn.train_top_layer(itertools.chain(dem_corpus, dem_corpus, dem_corpus, lpc_corpus, dem_corpus, lpc_corpus))
    dbn.save('.')

    test(lpc_corpus)

    dbn.add_layer(cRBM('layer1', 128, 9, 32,
            w_init=0.2, hb_init=-0.6, lr=1e-4, lrd=1e-4, wd=0.5, spt=0.1, spl=0.5,
            spe=0.1, mo=0.5, categorical_inputs=False))
    
    # dbn.load('.')
    dbn.train_top_layer(itertools.chain(dem_corpus, lpc_corpus, lpc_corpus, lpc_corpus))
    dbn.save('.')

    test(lpc_corpus)

