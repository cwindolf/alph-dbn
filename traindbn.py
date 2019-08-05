import numpy as np
import os.path
import tensorflow as tf
import string, itertools
import gensim.utils
from alphdbn import net
import matplotlib.pyplot as plt


seq_len = 128
n_hid = 2048
batch_size = 16
fw = 7
alphabet = string.ascii_lowercase + '\'\"., '
alphabet_size = len(alphabet)
itoa = dict(enumerate(alphabet))
atoi = dict((a, i) for i, a in enumerate(alphabet))

demons_txt = 'demons.txt'
lpc_txt = 'lpc.txt'

# -------------------------- data processing --------------------------

def munge(aloloc):
    aloi = []
    for aloc in aloloc:
        aloc = gensim.utils.deaccent(aloc)
        if aloc.strip():
            for c in ' '.join(aloc.lower().strip().split()):
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

    dbn = net(alphabet_size, seq_len, fw, n_hid, batch_size)
    init_op_ = tf.global_variables_initializer()

    with tf.Session() as sess:
        task = input('train/sample/restart: ')
        assert task in ('train', 'sample', 'restart')
        if task == 'restart':
            if os.path.exists('model.npz'):
                os.remove('model.npz')
            task = 'train'

        if os.path.exists('model.npz'):
                with np.load('model.npz') as model:
                    dbn.filters.load(model['filters'], sess)
                    dbn.biases.load(model['biases'], sess)

        if task == 'train':
            costs = []
            sess.run(init_op_)
            for i in range(100000):
                rand_starts = np.random.randint(dem_corpus.size - seq_len, size=batch_size)
                batch_getter = np.r_[('0,2', *tuple(slice(s, s + seq_len) for s in rand_starts))]
                cost, hm, _ = sess.run([dbn.cost, dbn.hid_mean, dbn.train_op], feed_dict={
                        dbn.beta: 0.5,
                        dbn.stimulus: dem_corpus[batch_getter],
                    })
                costs.append(cost)

                if not i % 1000:
                    print(cost, hm)
                    # run sample chain
                    samples = dem_corpus[batch_getter]
                    results = [samples]
                    for j in range(5):
                        samples = sess.run(dbn.vis_sample, feed_dict={
                                dbn.beta: 1.0,
                                dbn.stimulus: samples,
                            })
                        results.append(samples)
                    for j in range(batch_size):
                        print('start:')
                        for sample in results:
                            print(''.join(itoa[i] for i in sample[j]))

                    filters, biases = sess.run([dbn.filters, dbn.biases])
                    plt.subplot(121)
                    plt.hist(filters.ravel())
                    plt.subplot(122)
                    plt.hist(biases.ravel())
                    plt.savefig('hi.png')
                    plt.close('all')
                    plt.plot(costs)
                    plt.savefig('cs.png')

                    np.savez('model.npz', filters=filters, biases=biases)
        else:
            def beta_of_t(t):
                return 1 + np.sqrt(t) / 256  - (0.5 + np.cos(np.pi * t / 64) / 4) / np.log(t / 128 + 2)
            plt.plot(np.arange(1024), beta_of_t(np.arange(1024)))
            plt.show()
            while True:
                results = dbn.vis_sample_chain(sess, beta_of_t, 1024, 8)
                for j in range(results.shape[1]):
                    print('start:')
                    for sample in results:
                        print(''.join(itoa[i] for i in sample[j]))
