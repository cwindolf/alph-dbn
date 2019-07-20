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
    w, h_ = x.shape
    assert h == h_, f"need (x height:{h_})==(K in height: {h})"
    # wh -> 1hw1
    x = tf.transpose(x, (1, 0))
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, -1)
    # hfH -> hf1H
    K = tf.expand_dims(K, 2)
    # 1hw1, hf1H -> 1(h-h+1==1)(w-f+1)H
    res = tf.nn.conv2d(x, K, padding="VALID")
    # 11(w-f+1)H -> (w-f+1)H
    res = tf.squeeze(res)
    # let's make sure...
    assert res.shape == (w - f + 1, H)
    return res


def conv_adj(K, x):
    _, f, _ = K.shape
    return conv(adjoint(K), tf.pad(x, [(f - 1, f - 1), (0, 0)]))


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
        for i in self.odds:
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-2])
            else:
                state[i] = self.blocks[i](state[i - 1], state[i + 1])

        # Now the evens
        for i in self.evens:
            if i == self.n - 1:
                state[i] = self.blocks[i](state[-2])
            elif i == 0:
                state[i] = self.blocks[i](state[1])
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
        "cost",
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
    nhid = (seq_len - filter_len + 1) * hid_height

    # variables -------------------------------------------------------
    filters_rv = tfd.Normal(
        loc=tf.constant(np.zeros(filter_shape, dtype=np.float32)),
        scale=tf.constant(
            np.full(filter_shape, filter_scale, dtype=np.float32)
        ),
    )
    filters_ = tf.get_variable(
        "filters0",
        initializer=np.random.normal(size=filter_shape).astype(np.float32),
        trainable=False,
    )

    # params ----------------------------------------------------------
    beta_ = tf.placeholder(tf.float32, shape=(), name="beta")

    # samplers --------------------------------------------------------
    def vis_given_hid(hiddens_):
        print(f"vis_given_hid, hiddens_.shape={hiddens_.shape}")
        in_logits_ = beta_ * conv_adj(filters_, hiddens_)
        in_cat_rv = tfd.Categorical(logits=in_logits_)
        return in_cat_rv.sample(sample_shape=vis_shape)

    def hid_given_vis(visibles_):
        print(f"hid_given_vis:")
        print(f"    visibles_.shape={visibles_.shape}")
        print(f"    visibles_.dtype={visibles_.dtype}")
        onehots_ = tf.one_hot(visibles_, alphabet_size)
        print(f"    onehots_.shape={onehots_.shape}")
        print(f"    filters_.shape={filters_.shape}")
        hid_logits_ = beta_ * conv(filters_, onehots_)
        print(f"    hid_logits_.shape={hid_logits_.shape}")
        hid_ber_rv = tfd.Bernoulli(logits=hid_logits_, dtype=tf.float32)
        print(f"    hid_ber_rv.batch_shape={hid_ber_rv.batch_shape}")
        hid_sample_ = hid_ber_rv.sample()
        print(f"    hid_sample_.shape={hid_sample_.shape}")
        return hid_sample_

    # log likelihood --------------------------------------------------
    def energy(visibles_, hiddens_, filters_):
        print(f"energy:")
        print(f"    visibles_.shape={visibles_.shape}")
        print(f"    visibles_.dtype={visibles_.dtype}")
        print(f"    hiddens_.shape={hiddens_.shape}")
        print(f"    hiddens_.dtype={hiddens_.dtype}")
        print(f"    filters_.shape={filters_.shape}")
        print(f"    filters_.dtype={filters_.dtype}")
        onehots_ = tf.one_hot(visibles_, alphabet_size)
        print(f"    onehots_.shape={onehots_.shape}")
        print(f"    onehots_.dtype={onehots_.dtype}")
        in_logits_ = conv_adj(filters_, hiddens_)
        filter_energy_ = tf.reduce_sum(filters_rv.log_prob(filters_))
        return filter_energy_ + tf.reduce_sum(in_logits_ * onehots_)

    # training step ops -----------------------------------------------
    # pass in a stimulus...
    stimulus_ = tf.placeholder(tf.int32, shape=vis_shape, name="stimulus")
    # start with hids zeroed out
    init_hids_ = tf.zeros(hid_shape, name="init_hids")
    # get sample
    gk = GibbsKernel(
        energy, [vis_given_hid, hid_given_vis], extra_args=[filters_]
    )
    # This doesn't quite work yet -- some shape problem with mh's one_step.
    # mhk = tfp.mcmc.MetropolisHastings(inner_kernel=gk)
    res = tfp.mcmc.sample_chain(
        num_results=1,
        current_state=[stimulus_, init_hids_],
        kernel=gk,
        num_burnin_steps=100,
        trace_fn=None,
        parallel_iterations=1,
    )
    vis_sample_, hid_sample_ = res

    # hallucination ops -----------------------------------------------
    # start with random hids
    hid_rv = tfd.Bernoulli(
        probs=np.full(hid_shape, hid_p_target, dtype=np.float32),
        dtype=tf.float32,
    )
    hid_map_init_ = tf.reshape(hid_rv.sample(), (nhid,))

    def energy_and_grads(hiddens_):
        hiddens_ = tf.reshape(hiddens_, hid_shape)
        print("energy_and_grads:")
        energy_ = energy(stimulus_, hiddens_, filters_)
        print(f"    energy_.dtype={energy_.dtype}")
        print(f"    energy_.shape={energy_.shape}")
        grads_ = tf.gradients(energy_, hiddens_)[0]
        print(f"    grads_.shape={grads_.shape}")
        grads_ = tf.reshape(grads_, (nhid,))
        return energy_, grads_

    # get map estimate for hids
    hid_map_res = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=energy_and_grads,
        initial_position=hid_map_init_,
    )
    hid_map_ = tf.reshape(hid_map_res.position, hid_shape)
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
        cost=cost_,
        train_op=train_op_,
    )
    return net
