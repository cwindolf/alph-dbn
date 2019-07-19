import numpy as np
import tensorflow as tf
import string, itertools
from alphdbn import net


alphabet = string.ascii_lowercase + '., '
alphabet_size = len(alphabet)
itoa = dict(enumerate(alphabet))
atoi = dict((a, i) for i, a in enumerate(alphabet))

demons_txt = 'demons.txt'
lpc_txt = 'lpc.txt'

# *************************************************************************** #
# Data processing

def munge(aloloc, min_len=7):
    aloi = []
    for aloc in aloloc:
        aloi = []
        for c in aloc.strip().lower():
            if c in atoi:
                aloi.append(atoi[c])
        aloi.append(atoi[' '])
    return np.array(aloi, )


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

    # with open(LPC, 'r') as dat:
    #     lpc_corpus = munge(dat)

    with open(demons_txt, 'r') as dat:
        dem_corpus = munge(dat)

    # *********************************************************************** #
    # Train

    dbn = net(alphabet_size, 32, 5, 128)
    init_op_ = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op_)
        for i in range(10000):
            rand_start = np.random.randint(dem_corpus.size - 32)
            sess.run(dbn.train_op_, feed_dict={
                    dbn.beta: 0.4,
                    dbn.stimulus: dem_corpus[rand_start:rand_start + 32],
                })

            if not i % 100:
                # run sample chain
                sample = dem_corpus[rand_start:rand_start + 32]
                for j in range(20):
                    sample = sess.run(dbn.vis_sample, feed_dict={
                            dbn.beta: 0.4,
                            dbn.stimulus: sample,
                        })
                    print(sample)
