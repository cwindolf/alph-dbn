import numpy as np
import tensorflow as tf
import string, itertools
from alphdbn import net


seq_len = 128
alphabet = string.ascii_lowercase + '., '
alphabet_size = len(alphabet)
itoa = dict(enumerate(alphabet))
atoi = dict((a, i) for i, a in enumerate(alphabet))

demons_txt = 'demons.txt'
lpc_txt = 'lpc.txt'

# -------------------------- data processing --------------------------

def munge(aloloc, min_len=7):
    aloi = []
    for aloc in aloloc:
        for c in aloc.strip().lower():
            if c in atoi:
                aloi.append(atoi[c])
        aloi.append(atoi[' '])
    return np.array(aloi).astype(np.int32)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # ------------------------------ munge ----------------------------

    # with open(LPC, 'r') as dat:
    #     lpc_corpus = munge(dat)

    with open(demons_txt, 'r') as dat:
        dem_corpus = munge(dat)
    print('dem corpus', dem_corpus.dtype)
    print(dem_corpus)

    # ------------------------------ train ----------------------------

    dbn = net(alphabet_size, seq_len, 5, 128)
    init_op_ = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op_)
        for i in range(10000):
            rand_start = np.random.randint(dem_corpus.size - seq_len)
            cost, _ = sess.run([dbn.cost, dbn.train_op], feed_dict={
                    dbn.beta: 1.0,
                    dbn.stimulus: dem_corpus[rand_start:rand_start + seq_len],
                })
            print(cost)

            if not i % 250:
                # run sample chain
                sample = dem_corpus[rand_start:rand_start + seq_len]
                print('start:')
                print(''.join(itoa[i] for i in sample))
                for j in range(5):
                    sample, = sess.run(dbn.vis_sample, feed_dict={
                            dbn.beta: 2.0,
                            dbn.stimulus: sample,
                        })
                    print(''.join(itoa[i] for i in sample))
