import numpy as np
import scipy.signal as si
from scipy.special import expit as sigmoid

import functools
import os.path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class AlphDBN(object):

    def __init__(self, name, N):
        print('AlphDBN', name, N)
        self.name = name
        self.N = N
        self.eye = np.eye(self.N)
        self.layers = []

        if not os.path.exists(self.name):
            os.mkdir(self.name)

    def data_gen_for_top_layer(self, seqs):
        for seq in seqs:
            res = self.eye[seq]
            for layer in self.layers[:-1]:
                res = layer.hid_means(res)
            yield res

    def add_layer(self, layer):
        self.layers.append(layer)

    def train_top_layer(self, alph_seqs):
        top_layer = self.layers[-1]
        for example in self.data_gen_for_top_layer(alph_seqs):
            top_layer.learn_from_example(example)

    def save(self, path):
        print('Saving dbn.')
        for i, layer in enumerate(self.layers):
            layer.save(os.path.join(path, self.name))

    def load(self, path):
        print('dbn loading', len(self.layers), 'layers.')
        for i, layer in enumerate(self.layers):
            layer.load(os.path.join(path, self.name))

    def gibbs(self, start_vis_batch, n_gibbs, beta=1.0, beta_rate=1.0,
              max_beta=2.0, yield_samples=True):
        # step 0
        activities = [self.eye[start_vis_batch]]
        for i, layer in enumerate(self.layers):
            activities.append(layer.hid_means(activities[i], beta=beta))

        # Step k
        for k in range(n_gibbs):
            beta = min(beta_rate * beta, max_beta)
            for i in range(len(self.layers) - 2, -1, -1):
                activities[i + 1] = self.layers[i].hid_means(
                    activities[i],
                    topdown=self.layers[i + 1].vis_x(activities[i + 2],
                                                     beta=beta),
                    beta=beta)

            s, activities[0] = self.layers[0].sample_vis(
                activities[1], return_both=True, beta=beta)

            for i in range(0, len(self.layers) - 1, 1):
                activities[i + 1] = self.layers[i].hid_means(
                    activities[i],
                    topdown=self.layers[i + 1].vis_x(activities[i + 2],
                                                     beta=beta),
                    beta=beta)

            activities[-1] = self.layers[-1].hid_means(
                activities[-2], beta=beta)

            if yield_samples:
                yield s
            else:
                yield activities


class CategoricalFullHeightConvRBM(object):

    def __init__(self, name, in_height, kernel_width, out_height, spars_dict,
                 w_init=0.05, hb_init=-0.6, lr=1e-3, lrd=0.0, wd=2.0, mo=0.5,
                 categorical_inputs=True, plot_every=1000, itoa=None, pad_i=None):
        # print('New RBM', name, "in_height", in_height, "kernel_width", kernel_width, "out_height", out_height,
        #     "w_init", w_init, "hb_init", hb_init, "lr", lr, "lrd", lrd, "wd", wd, "spt", spt, "spl", spl,
        #     "spe", spe, "mo", mo)
        self.name = name
        self.in_height = in_height
        self.out_height = out_height
        self.kernel_size = (out_height, kernel_width, in_height)
        self.pad_width = kernel_width - 1
        self.kernels = np.random.uniform(
            low=-w_init, high=w_init, size=self.kernel_size)
        # self.kernels = np.random.triangular(-w_init, 0.0, w_init, size=self.kernel_size)
        self.v_bias = 0.0
        self.h_bias = np.full(out_height, hb_init)
        self.lr = lr
        self.lrd = lrd
        self.wd = wd
        self.spars_dict = spars_dict
        self.running_mean_activity = np.zeros(out_height)
        self.mo = mo
        self.kmo = np.zeros_like(self.kernels)
        self.vbmo = 0.0
        self.hbmo = np.zeros(out_height)
        self.categorical_inputs = categorical_inputs
        self.plot_every = plot_every
        self.itoa = itoa
        self.pad_i = pad_i
        self._step = 0
        self._fig, self._axes = None, None

    @property
    def adjoint_kernels(self):
        return np.flip(self.kernels, axis=1).transpose(2, 1, 0)

    def _pad_hid(self, z_H):
        return np.pad(z_H, ((self.pad_width, self.pad_width), (0, 0)), 'constant')

    def _pad_vis(self, z_H):
        pads = z_H
        # pads = np.pad(z_H, ((self.pad_width, self.pad_width), (0, 0)), 'constant')
        # if self.pad_i is not None:
        #     pads[0:self.pad_width, self.pad_i] = 1.0
        #     pads[-self.pad_width:, self.pad_i] = 1.0
        return pads

    def vis_x(self, z_H, beta=1.0):
        return beta * (self.fh_conv2d(self._pad_hid(z_H), self.adjoint_kernels) + self.v_bias)

    def vis_means(self, z_H, beta=1.0):
        logits = self.vis_x(z_H, beta=beta)
        if self.categorical_inputs:
            return self.csoftmax(logits)
        else:
            return sigmoid(logits)

    def hid_x(self, z_V, beta=1.0):
        return beta * (self.fh_conv2d(self._pad_vis(z_V), self.kernels) + self.h_bias)

    def hid_means(self, z_V, topdown=0.0, beta=1.0):
        return sigmoid(self.hid_x(z_V, beta=beta) + topdown)

    def sample_vis(self, z_H, beta=1.0, return_both=False):
        probs = self.vis_means(z_H, beta=beta)
        if self.categorical_inputs:
            samples = np.empty_like(probs)
            for i in range(probs.shape[0]):
                samples[i, :] = np.random.multinomial(1, probs[i])
        else:
            samples = np.random.binomial(1, probs)
        if return_both:
            return samples, probs
        return samples

    def sample_hid(self, z_V, topdown=0.0, beta=1.0, return_both=False):
        probs = self.hid_means(z_V, topdown=topdown, beta=beta)
        samples = np.random.binomial(1, probs)
        if return_both:
            return samples, probs
        return samples

    def learn_from_example(self, example, beta=1.0):
        self.lr *= 1 - self.lrd

        # Gibbs chain
        h0 = self.hid_means(example, beta=beta)
        v1 = self.vis_means(h0, beta=beta)
        h1 = self.hid_means(v1, beta=beta)

        # Gradients
        gK = np.flip(self.fh_conv2d_transpose(example, h0) -
                     self.fh_conv2d_transpose(v1, h1), axis=1)
        # Will be 0 in a net with categorical inputs.
        gV = (example - v1).sum() / self.in_height
        gH = (h0 - h1).sum(axis=0)

        # Sparsity regularization
        if self.spars_dict['type'] == 0:
            self.running_mean_activity = self.spars_dict['eta'] * h0.mean(
                axis=0) + (1 - self.spars_dict['eta']) * self.running_mean_activity
            sp_H = self.spars_dict['lambda'] * \
                (self.spars_dict['target'] - self.running_mean_activity)
            sp_K = sp_H[..., None, None]
        elif self.spars_dict['type'] == 1:
            sp_H = self.spars_dict['lambda'] * \
                ((1.0 - h0) * h0 *
                 (self.spars_dict['target'] - h0)).sum(axis=0)
            sp_K = self.spars_dict['lambda'] * self.fh_conv2d_transpose(
                example, (1.0 - h0) * h0 * (self.spars_dict['target'] - h0))

        # The updates
        dK = self.kmo + self.lr * (gK - self.wd * self.kernels + sp_K)
        dV = self.vbmo + self.lr * gV
        dH = self.hbmo + self.lr * (gH + sp_H)

        self.kernels += dK
        self.v_bias += dV
        self.h_bias += dH

        self.kmo = self.mo * dK
        self.vbmo = self.mo * dV
        self.hbmo = self.mo * dH

        if not self._step % self.plot_every:
            print(self._step, 'LR', self.lr, 'Hmean', h0.mean(),
                  'fmin,fmax:', self.kernels.min(), self.kernels.max())
            self.plot(example, h0, v1, h1, gK, gV, gH)

        self._step += 1

    def plot(self, v0, h0, v1, h1, gK, gV, gH):
        if self._fig is None:
            self._fig, self._axes = plt.subplots(4, 2, figsize=(10, 10))

        for ac, ax in zip((v0, h0, v1, h1), self._axes[0:2].flat):
            ax.imshow(ac, interpolation='nearest')

        ker = self.kernels[np.random.randint(self.out_height)]
        self._axes[2, 0].imshow(ker)
        if self.itoa is not None:
            self._axes[2, 0].set_xlabel(
                ''.join(self.itoa[i] for i in ker.argmax(axis=1)))
        self._axes[2, 1].hist(gK.flat)
        self._axes[3, 0].hist([self.h_bias, self.v_bias], density=True)
        self._axes[3, 1].hist([gH, gV], density=True)

        self._fig.savefig('%s%06d.png' % (self.name, self._step),
                          bbox_inches='tight', pad_inches=0)

        for ax in self._axes.flat:
            ax.clear()

    @staticmethod
    def softmax(X, axis=None):
        z = np.atleast_2d(X)
        z = z - z.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        return p.flatten() if X.ndim == 1 else p
    csoftmax = functools.partialmethod(softmax, axis=1)

    @staticmethod
    def fh_conv2d(X, kernel):
        '''
        no batch dim! go use tensorflow.
        convolutions specialized when the kernel is full height, and when your
        inputs always have 1 channel. (in these models, height takes the role
        of channels.)

        X       WH
        kernel  OwH

        output  (W-w+1)O
        '''
        output = np.empty((X.shape[0] - kernel.shape[1] + 1, kernel.shape[0]))
        for o, k_o in enumerate(kernel):
            output[:, o] = si.correlate2d(X, k_o, 'valid').squeeze()
        return output

    @staticmethod
    def fh_conv2d_transpose(X, Y):
        '''
        transpose of fh_conv2d for computing weight gradients/unit correlations
        in the model.

        X       WH
        Y       (W-w+1)O

        output  OwH
        '''
        output = np.empty(
            (Y.shape[1], X.shape[0] - Y.shape[0] + 1, X.shape[1]))
        for o, Y_o in enumerate(Y.T):
            output[o, :, :] = si.correlate2d(X, Y_o[..., None], 'valid')
        return output

    def save(self, path):
        np.savez_compressed(os.path.join(path, self.name + '.npz'),
                            k=self.kernels, v=self.v_bias, h=self.h_bias)

    def load(self, path):
        dat = np.load(os.path.join(path, self.name + '.npz'))
        self.kernels = dat['k']
        self.v_bias = dat['v']
        self.h_bias = dat['h']
