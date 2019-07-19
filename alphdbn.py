import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple

tfd = tfp.distributions


# ------------------------------- lib ---------------------------------


def adjoint(filters):
    return tf.transpose(tf.reverse(filters, axis=1), (2, 1, 0))


def conv(K, x):
    """strided on width, fixed on height"""
    # hw -> 1hw1
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, -1)
    # hfH -> 1hfH
    K = tf.expand_dims(K, 0)
    # 11(w-f+1)H -> H(w-f+1)
    res = tf.nn.conv2d(x, K, padding="VALID")
    res = tf.squeeze(res)
    res = tf.transpose(res, (1, 0))
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
        self.odds = list(range(1, self.n, 2))
        self.evens = list(range(2, self.n, 2))
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
        for i in range(self.odds):
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-1])
            else:
                state[i] = self.blocks[i](state[i - 1], state[i + 1])

        # Now the evens
        for i in range(self.evens):
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-1])
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
        "vis_sample",
        "hid_sample",
        "train_op",
    ],
)


def net(
    alphabet_size,
    seq_len,
    filter_len,
    hid_height,
    hid_p_target=0.1,
    filter_scale=1.0,
    lr=0.01,
    momentum=0.1,
):
    """Make an RBM with categorical inputs
    """
    # shapes ----------------------------------------------------------
    vis_shape = (seq_len,)
    filter_shape = (alphabet_size, filter_len, hid_height)
    hid_shape = (seq_len - filter_len + 1, hid_height)

    # variables -------------------------------------------------------
    filters_rv = tfd.Normal(
        loc=tf.constant(np.zeros(filter_shape, dtype=np.float32)),
        scale=tf.constant(
            np.full(filter_shape, filter_scale, dtype=np.float32)
        ),
    )
    filters_ = tf.get_variable(
        "filters0", initializer=np.random.normal(size=filter_shape)
    )

    # params ----------------------------------------------------------
    beta_ = tf.placeholder(tf.float32, name="beta")

    # samplers --------------------------------------------------------
    def vis_given_hid(hiddens_):
        in_logits_ = beta_ * conv(adjoint(filters_), hiddens_)
        in_cat_rv_ = tfd.Categorical(logits=in_logits_)
        return in_cat_rv_.sample()

    def hid_given_vis(visibles_):
        onehots_ = tf.one_hot(visibles_, alphabet_size)
        hid_logits_ = beta_ * conv(filters_, onehots_)
        hid_ber_rv_ = tfd.Bernoulli(logits=hid_logits_)
        return hid_ber_rv_.sample()

    # log likelihood --------------------------------------------------
    def energy(inputs_, hiddens_, filters_):
        in_logits_ = conv(adjoint(filters_), hiddens_)
        filter_energy_ = filters_rv.log_prob(filters_)
        return filter_energy_ + tf.reduce_sum(in_logits_ * inputs_)

    # training step ops -----------------------------------------------
    # pass in a stimulus...
    stimulus_ = tf.placeholder(tf.int32, shape=vis_shape, name="stimulus")
    # start with hids zeroed out
    init_hids_ = tf.zeros(hid_shape, name="init_hids")
    # get sample
    mhk = tfp.mcmc.MetropolisHastings(
        inner_kernel=GibbsKernel(
            energy, [vis_given_hid, hid_given_vis], extra_args=[filters_]
        )
    )
    vis_sample_, hid_sample_ = tfp.mcmc.sample_chain(
        num_results=1,
        current_state=[stimulus_, init_hids_],
        kernel=mhk,
        num_burnin_steps=10,
    )

    # hallucination ops -----------------------------------------------
    # start with random hids
    hid_rv = tfd.Bernoulli(
        probs=np.full(hid_shape, hid_p_target, dtype=np.float32)
    )
    hid_map_init_ = hid_rv.sample()
    # get map estimate for hids
    def energy_and_grads(hiddens_):
        energy_ = energy(stimulus_, hiddens_, filters_)
        grads_ = tf.gradients(energy_, hiddens_)
        return energy_, grads_

    hid_map_res = tfp.optimizer.lbfgs_minimize(
        energy_and_grads, hid_map_init_, max_iterations=100
    )
    hid_map_ = hid_map_res.position
    # do some gradient descent on filters with vis/hids fixed
    opt = tf.train.MomentumOptimizer(lr, momentum)
    cost_ = energy(stimulus_, hid_map_, filters_)
    train_op_ = opt.minimize(cost_, var_list=[filters_])

    # build return pack -----------------------------------------------
    net = AlphRBM(
        vis_shape=vis_shape,
        filter_shape=filter_shape,
        hid_shape=hid_shape,
        beta=beta_,
        stimulus=stimulus_,
        filters=filters_,
        vis_sample=vis_sample_,
        hid_sample=hid_sample_,
        train_op=train_op_,
    )
    return net
