import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple

tfd = tfp.distributions


# ------------------------------- lib ---------------------------------


def adjoint(filters):
    return tf.transpose(tf.reverse(filters, axis=(1,)), (2, 1, 0))


def conv(K, x):
    """strided on width, fixed on height"""
    h, f, H = K.shape
    print(x)
    b, w, h_ = x.shape
    print(b, w, h_)
    assert h == h_, f"need (x height:{h_})==(K in height: {h})"
    # bwh -> bhw1
    x = tf.transpose(x, (0, 2, 1))
    x = tf.expand_dims(x, -1)
    # hfH -> hf1H
    K = tf.expand_dims(K, 2)
    # bhw1, hf1H -> b(h-h+1==1)(w-f+1)H
    res = tf.nn.conv2d(x, K, padding="VALID")
    # b1(w-f+1)H -> b(w-f+1)H
    res = tf.squeeze(res)
    # let's make sure...
    assert res.shape == (b, w - f + 1, H)
    return res


def conv_adj(K, x):
    _, f, _ = K.shape
    res = conv(adjoint(K), tf.pad(x, [(0, 0), (f - 1, f - 1), (0, 0)]))
    print(f'conv_adj: K.shape={K.shape}, x.shape={x.shape}, res.shape={res.shape}')
    return res


KernelResults = namedtuple("KernelResults", ["target_log_prob"])


class GibbsKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, energy, blocks, extra_args=[]):
        """
        energy: callable
            receives samples from blocks and returns the energy
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
        """
        self.n = len(blocks)
        print(f"GibbsKernel n={self.n}")
        self.odds = list(range(1, self.n, 2))
        self.evens = list(range(0, self.n, 2))
        print(self.odds, self.evens)
        self.blocks = blocks
        self.energy = energy
        self.extra_args = extra_args

    @property
    def is_calibrated(self):
        return False

    def bootstrap_results(self, init_state):
        return KernelResults(self.energy(*init_state, *self.extra_args))

    def one_step(self, current_state, previous_kernel_results):
        state = [s for s in current_state]  # shallow copy
        # Sample odd layers first.
        for i in self.odds:
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-2])
            else:
                state[i] = self.blocks[i](state[i - 1], state[i + 1])

        # Now the evens
        for i in self.evens:
            if i == 0:
                state[i] = self.blocks[i](state[1])
            elif i == self.n - 1:
                state[i] = self.blocks[i](state[-2])
            else:
                state[i] = self.blocks[i](state[i - 1], state[i + 1])

        return [state, KernelResults(self.energy(*state, *self.extra_args))]


# -------------------------------- net --------------------------------


AlphRBM = namedtuple(
    "AlphRBM",
    [
        "vis_shape",
        "filter_shape",
        "hid_shape",
        "beta",
        "stimulus",
        "filters",
        "biases",
        "vis_sample",
        "hid_sample",
        "hid_mean",
        "cost",
        "train_op",
        "vis_sample_chain",
    ],
)


def net(
    alphabet_size,
    seq_len,
    filter_len,
    hid_height,
    batch_size,
    sparse_hid=0.0,
    lr=5e-5,
    momentum=0.01,
    wd=1e-5,
):
    """Make an RBM with categorical inputs
    """
    # shapes ----------------------------------------------------------
    vis_shape = (batch_size, seq_len,)
    filter_shape = (alphabet_size, filter_len, hid_height)
    hid_shape = (batch_size, seq_len - filter_len + 1, hid_height)
    nhid = (seq_len - filter_len + 1) * hid_height

    # variables -------------------------------------------------------
    filters_ = tf.get_variable(
        "filters0",
        initializer=np.random.normal(scale=0.1, size=filter_shape).astype(np.float32),
        trainable=False,
    )
    biases_ = tf.get_variable(
        "biases0",
        initializer=np.random.normal(scale=0.1, size=hid_shape).astype(np.float32),
        trainable=False,
    )

    # params ----------------------------------------------------------
    beta_ = tf.placeholder(tf.float32, shape=(), name="beta")

    # samplers --------------------------------------------------------
    def vis_given_hid(hiddens_):
        in_logits_ = beta_ * conv_adj(filters_, hiddens_)
        print('vis_given_hid, in_logits_', in_logits_.shape)
        in_cat_rv = tfd.Categorical(logits=in_logits_)
        vis_sample_ = in_cat_rv.sample()
        print('vis_given_hid, vis_sample_', vis_sample_.shape)
        return vis_sample_

    def hid_given_vis(visibles_):
        onehots_ = tf.one_hot(visibles_, alphabet_size)
        hid_logits_ = beta_ * (conv(filters_, onehots_) + biases_)
        hid_ber_rv = tfd.Bernoulli(logits=hid_logits_, dtype=tf.float32)
        hid_sample_ = hid_ber_rv.sample()
        return hid_sample_

    # log likelihoods -------------------------------------------------
    def energy(visibles_, hiddens_, filters_, biases_):
        onehots_ = tf.one_hot(visibles_, alphabet_size)
        hid_logits_ = beta_ * (conv(filters_, onehots_) + biases_)
        model_energy_ = -tf.reduce_sum(hid_logits_ * hiddens_)
        # filter_energy_ = wd * tf.reduce_sum(tf.square(filters_))
        # sparsity_energy_ = sparse_hid * tf.reduce_sum(tf.abs(hiddens_))
        # print_energies_ = tf.print(
        #     "model_energy_", model_energy_,
        #     "filter_energy_", filter_energy_)
        # with tf.control_dependencies([print_energies_]):
        total_energy = model_energy_
        return total_energy

    def free_energy_vis(visibles_, filters_, biases_):
        return -tf.reduce_sum(
            tf.log(1.0
                   + tf.exp(
                    beta_ * (
                        biases_
                        + conv(
                            filters_,
                            tf.one_hot(visibles_, alphabet_size))))))

    # hallucination ops -----------------------------------------------
    # pass in a stimulus...
    stimulus_ = tf.placeholder(tf.int32, shape=vis_shape, name="stimulus")
    # start with hids zeroed out
    init_hids_ = tf.zeros(hid_shape, name="init_hids")
    # get sample
    gk = GibbsKernel(
        energy, [vis_given_hid, hid_given_vis], extra_args=[filters_, biases_]
    )
    # This doesn't quite work yet -- some shape problem with mh's one_step.
    # mhk = tfp.mcmc.MetropolisHastings(inner_kernel=gk)
    res = tfp.mcmc.sample_chain(
        num_results=1,
        current_state=[stimulus_, init_hids_],
        kernel=gk,
        num_burnin_steps=0,
        trace_fn=None,
        parallel_iterations=1,
    )
    long_chain_res = tfp.mcmc.sample_chain(
        num_results=128,
        current_state=[stimulus_, init_hids_],
        kernel=gk,
        num_burnin_steps=0,
        num_steps_between_results=2,
        trace_fn=None,
        parallel_iterations=1)
    print('lcr', long_chain_res)
    long_chain_vis = tf.squeeze(long_chain_res[0])
    print('mcmcres', res)
    vis_sample_, hid_sample_ = (tf.squeeze(s) for s in res)

    print('mcmcres vis_sample_', vis_sample_)

    # training ops ----------------------------------------------------
    # sample
    logits_ = beta_ * conv(filters_, tf.one_hot(stimulus_, alphabet_size))
    hid_rv = tfd.Bernoulli(
        logits=logits_,
        dtype=tf.float32,
    )
    hid_sample_ = hid_rv.sample()
    hid_mean_ = tf.reduce_mean(hid_sample_)
    # do some gradient descent on filters with vis/hids fixed
    opt = tf.train.MomentumOptimizer(lr, momentum)
    cost_ = (free_energy_vis(stimulus_, filters_, biases_)
            - free_energy_vis(vis_sample_, filters_, biases_)) / batch_size
    cost_ = cost_ + wd * (tf.reduce_sum(tf.square(filters_))
                         + tf.reduce_sum(tf.square(biases_)))
    train_op_ = opt.minimize(cost_, var_list=[filters_, biases_])

    # hallucination function ------------------------------------------
    def vis_sample_chain(sess, beta=1.0):
        start_vis = np.random.randint(alphabet_size, size=(batch_size, seq_len))
        vis_samples = sess.run(long_chain_vis, feed_dict={
                stimulus_: start_vis,
                beta_: beta,
            })
        return vis_samples

    # build return pack -----------------------------------------------
    net = AlphRBM(
        vis_shape=vis_shape,
        filter_shape=filter_shape,
        hid_shape=hid_shape,
        beta=beta_,
        stimulus=stimulus_,
        filters=filters_,
        biases=biases_,
        vis_sample=vis_sample_,
        hid_sample=hid_sample_,
        cost=cost_,
        hid_mean=hid_mean_,
        train_op=train_op_,
        vis_sample_chain=vis_sample_chain,
    )
    return net
