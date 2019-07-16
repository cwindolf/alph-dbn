import numpy as np
# import tensorflow as tf
import tensorflow.probability as tfp

tfd = tfp.distributions

# Model:
#  - inputs:
#       ()
#  - filters:
#
#  - hiddens:
#


# ------------------------------- lib ---------------------------------

def adjoint(filters):
    return tf.transpose(
        tf.reverse(filters, axis=1),
        (2, 1, 0))


def conv(K, x):
    '''strided on width, fixed on height'''
    # hw -> 1hw1
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, -1)
    # hfH -> 1hfH
    K = tf.expand_dims(K, 0)
    # 11(w-f+1)H -> H(w-f+1)
    res = tf.nn.conv2d(
        x,
        K,
        padding='VALID')
    res = tf.squeeze(res)
    res = tf.transpose(res, (1, 0))
    return res


class GibbsKernel(tfp.mcmc.TransitionKernel):

    def __init__(self, blocks):
        '''
        blocks:
            list of functions such that blocks[i] generates samples
            from the condtional distribution
                blocks[i] | blocks[i - 1], blocks[i + 1]
            this assumes that the model factors so that each block
            is conditionally independent from the rest given the
            one before and the one after (except for the first and
            last block...)
            then we can alternate sampling the even and odd layers.
            blocks[0] and blocks[n - 1] should accept one tensor of samples.
            blocks[1] through blocks[n - 2] should accept two tensors, the
            first from below and the second from above.
        '''
        self.n = len(blocks)
        self.odds = list(range(1, n, 2))
        self.evens = list(range(2, n, 2))
        self.blocks = blocks

    def bootstrap_results(self, init_state):
        return ()

    def one_step(self, current_state, previous_kernel_results):
        state = [s for s in current_state]  # shallow copy
        # Sample odd layers first.
        odd_samples = []
        for i in range(self.odds):
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-1])
            else:
                state[i] = self.blocks[i](state[i - 1], state[i + 1])

        # Now the evens
        even_samples = [self.blocks[0](current_state[1])]
        for i in range(evens):
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-1])
            else:
                state[i] = self.blocks[i](state[i - 1], state[i + 1])

        return [state, ()]


# ---------------------------- first layer ----------------------------


def cat_layer_graph(alphabet_size, seq_len, filter_len, hid_height,
                    hid_p_target=0.1, filter_scale=1.0):
    # shapes ----------------------------------------------------------
    in_shape = (seq_len, alphabet_size)
    filter_shape = (alphabet_size, filter_len, hid_height)
    out_shape = (seq_len - filter_len + 1, hid_height)

    # params ----------------------------------------------------------
    beta = tf.placeholder(tf.float32, name='beta')

    # rvs -------------------------------------------------------------
    hiddens_rv = tfd.Bernoulli(
        logits=np.full(out_shape, np.log(hid_p_target), dtype=np.float32),
        name='layer1_hiddens_rv')
    filters_rv = tfd.Normal(
        loc=np.full(filter_shape, 0.0, dtype=np.float32),
        scale=np.full(filter_shape, filter_scale, dtype=np.float32),
        name='layer1_filters_rv')
    inputs_rv = tfd.Categorical(
            logits=)

    def energy(inputs_, hiddens_, filters_):
        in_logits_ = conv(adjoint(filters_), hiddens_)
        return tf.reduce_sum(in_logits_ * inputs_)
