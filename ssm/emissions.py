from warnings import warn
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import hessian

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
from ssm.stats import independent_studentst_logpdf, bernoulli_logpdf
from ssm.regression import fit_linear_regression, fit_constrained_linear_regression
from scipy.stats import invwishart
import ssm.stats as stats

# Observation models for SLDS
class Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self,):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplementedError

    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def sample(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        warn("Analytical Hessian is not implemented for this Emissions class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        obj = lambda xt, datat, inputt, maskt: \
            self.log_likelihoods(datat[None,:], inputt[None,:], maskt[None,:], tag, xt[None,:])[0, 0]
        hess = hessian(obj)
        terms = np.array([np.squeeze(hess(xt, datat, inputt, maskt))
                          for xt, datat, inputt, maskt in zip(x, data, input, mask)])
        return -1 * terms

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, emission_block_diagonal = False, 
               **kwargs):
        """
        If M-step in Laplace-EM cannot be done in closed form for the emissions, default to SGD.
        """

        if emission_block_diagonal>0:
            raise ValueError("Block diagonal emissions not implemented for this Emissions class.")

        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log likelihood
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = 0
            obj += self.log_prior()
            for data, input, mask, tag, x, (Ez, _, _) in \
                zip(datas, inputs, masks, tags, continuous_expectations, discrete_expectations):
                obj += np.sum(Ez * self.log_likelihoods(data, input, mask, tag, x))
            return -obj / T

        # Optimize emissions log-likelihood
        self.params = optimizer(_objective, self.params,
                                num_iters=maxiter,
                                suppress_warnings=True,
                                **kwargs)


# Many emissions models start with a linear layer
class _LinearEmissions(Emissions):
    """
    A simple linear mapping from continuous states x to data y.

        E[y | x] = Cx + d + Fu

    where C is an emission matrix, d is a bias, F an input matrix,
    and u is an input.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer.  Set _Cs to be private so that it can be
        # changed in subclasses.
        self._Cs = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        K, N, D = self.K, self.N, self.D
        assert value.shape == (1, N, D) if self.single_subspace else (K, N, D)
        self._Cs = value

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # Invert with the average emission parameters
        C = np.mean(self.Cs, axis=0)
        F = np.mean(self.Fs, axis=0)
        d = np.mean(self.ds, axis=0)
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
            + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
            + self.ds

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data with the maximum effective dimension
        pca, xs, ll = pca_with_imputation(min(self.D * Keff, self.N),
                                          resids, masks, num_iters=num_iters)

        # Assign each state a random projection of these dimensions
        Cs, ds = [], []
        for k in range(Keff):
            weights = npr.randn(self.D, self.D * Keff)
            weights = np.linalg.svd(weights, full_matrices=False)[2]
            Cs.append((weights @ pca.components_).T)
            ds.append(pca.mean_)

        # Find the components with the largest power
        self.Cs = np.array(Cs)
        self.ds = np.array(ds)

        return pca


class _OrthogonalLinearEmissions(_LinearEmissions):
    """
    A linear emissions matrix constrained such that the emissions matrix
    is orthogonal. Use the rational Cayley transform to parameterize
    the set of orthogonal emission matrices. See
    https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
    for a derivation of the rational Cayley transform.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_OrthogonalLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        assert N > D
        self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        for k in range(C0.shape[0]):
            C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        self.Cs = C0

    @property
    def Cs(self):
        D = self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
        Fs = np.matmul(T(self._As), self._As) - Bs
        trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
        trm2 = np.eye(D) + Fs
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(
            np.matmul(T(Cs), Cs),
            np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
            )
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.N, self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        # Make sure value is the right shape and orthogonal
        Keff = 1 if self.single_subspace else self.K
        assert value.shape == (Keff, N, D)
        assert np.allclose(
            np.matmul(T(value), value),
            np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
            )

        Q1s, Q2s = value[:, :D, :], value[:, D:, :]
        Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._Ms = T(Fs)
        self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
        assert np.allclose(self.Cs, value)

    @property
    def params(self):
        return self._As, self._Ms, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self._As, self._Ms, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self._As = self._As[perm]
            self._Ms = self._Ms[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]


# Sometimes we just want a bit of additive noise on the observations
class _IdentityEmissions(Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_IdentityEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        assert N == D

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def forward(self, x, input, tag):
        return x[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is just the data
        """
        return np.copy(data)


# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_NeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D + M,) + hidden_layer_sizes + (N,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))
        for W, b in zip(self.weights, self.biases):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.D)


# Observation models for SLDS
class _GaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        # self.inv_etas = -1 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = np.random.randn(1, N, N) if single_subspace else np.random.randn(K, N, N)
        # parameters of the inverse wishart prior on the emission noise
        self.Psi0 = np.ones(1) if single_subspace else np.ones(K)
        self.nu0 = np.ones(1) if single_subspace else np.ones(K)

    @property
    def params(self):
        return super(_GaussianEmissionsMixin, self).params + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_GaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        mus = self.forward(x, input, tag).reshape((-1, data.shape[0], self.N))
        Sigmas = self.inv_etas

        # # TODO: this is wrong, needs to be fixed
        # lls = -0.5 * np.linalg.slogdet(2 * np.pi * etas)[1] - 0.5 * (data[:, None, :] - mus) @ np.linalg.inv(etas) @ ((data[:, None, :] - mus).transpose((0, 2, 1)))
        return np.column_stack([stats.multivariate_normal_logpdf(data, mu, Sigma)
                               for mu, Sigma in zip(mus, Sigmas)])
        # return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = self.inv_etas        
        samples_predicted = np.zeros((T, self.N))
        for t in range(T):
            samples_predicted[t] = npr.multivariate_normal(mus[t, z[t], :].reshape((self.N,)), etas[z[t]])
            # samples_predicted[t] = mus[t, z[t], :].reshape((self.N,))
        return samples_predicted

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        yhat = mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)
        return yhat
    
    def _compute_statistics_for_m_step(self, ys, inputs, masks, tags, continuous_expectations, discrete_expectations):
        """ compute all statistics needed for M step when a closed-form update is feasible"""
        K = self.K
        EyyT = np.zeros((K, self.N, self.N))
        ExxT = np.zeros((K, self.D+1, self.D+1))
        ExyT = np.zeros((K, self.D+1, self.N))
        weight_sum = 0

        weights = [np.ones(y.shape[0]) for y in ys]
        for y, input, weight, (_, Ex, smoothed_sigmas, _), (Ez, _, _), in zip(ys, inputs, weights, continuous_expectations, discrete_expectations):
            for k in range(self.K):
                w = Ez[:,k]
                EyyT[k] += np.einsum('t,ti,tj->ij',w, y, y)
                # ExxT     
                mumuT = np.einsum('ti,tj->tij',Ex, Ex) + smoothed_sigmas
                ExxT[k, :self.D,:self.D] += np.einsum('t, tij->ij',w, mumuT)
                ExxT[k, -1,:self.D] += np.einsum('t, ti->i',w, Ex)
                ExxT[k, :self.D, -1] += np.einsum('t, ti->i',w, Ex)
                ExxT[k, -1, -1] += np.sum(w)
                # ExyT
                ExyT[k, :self.D,:] += np.einsum('t,ti,tj->ij', w, Ex, y)
                ExyT[k, -1,:] += np.einsum('t,ti->i', w, y)
                weight_sum += np.sum(weight)
        return ExxT, ExyT, EyyT, weight_sum


class GaussianEmissions(_GaussianEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = pca.noise_variance_ + 1e-4*np.eye(self.N)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # TODO: check if things need to be changed here?
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            R_inv = np.linalg.inv(self.inv_etas[0])
            block = -1.0 * self.Cs[0].T@R_inv@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@inv_eta@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess
    
    def log_prior(self,):
        """ computes the log prior of the emission parameters """
        log_prior = 0
        K = 1 if self.single_subspace else self.K
        for k in range(K):
            # compute inverse wishart prior on the emission noise
            log_prior += log_prior + invwishart.logpdf(self.inv_etas[k], self.nu0[k] + self.N +1, self.Psi0[k]*np.eye(self.N))
        return log_prior

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):
        Xs = [np.column_stack([x, u]) for x, u in
              zip(continuous_expectations, inputs)]
        ys = datas
        ExxT, ExyT, EyyT, weight_sum = self._compute_statistics_for_m_step(ys, inputs, masks, tags, continuous_expectations, discrete_expectations)
        ws = [Ez for (Ez, _, _) in discrete_expectations]

        if self.single_subspace and all([np.all(mask) for mask in masks]):
            # Return exact m-step updates for C, F, d, and inv_etas
            # get expectations in right shape
            expectations = [ExxT[0], ExyT[0], EyyT[0], weight_sum]
            CF, d, Sigma = fit_linear_regression(
                Xs, ys,
                expectations=expectations,
                prior_ExxT=1e-4 * np.eye(self.D + self.M + 1),
                prior_ExyT=np.zeros((self.D + self.M + 1, self.N)))
            self.Cs = CF[None, :, :self.D]
            self.Fs = CF[None, :, self.D:]
            self.ds = d[None, :]
            self.inv_etas = np.log(np.diag(Sigma))[None, :]
        else:
            Cs, Fs, ds, inv_etas = [], [], [], []
            for k in range(self.K):
                expectations = [ExxT[k], ExyT[k], EyyT[k], weight_sum]
                CF, d, Sigma = fit_linear_regression(
                    Xs, ys, 
                    expectations=expectations, weights=[w[:, k] for w in ws],
                    prior_ExxT=1e-4 * np.eye(self.D + self.M + 1),
                    prior_ExyT=np.zeros((self.D + self.M + 1, self.N)))
                Cs.append(CF[:, :self.D])
                Fs.append(CF[:, self.D:])
                ds.append(d)
                inv_etas.append(np.log(np.diag(Sigma)))
            self.Cs = np.array(Cs)
            self.Fs = np.array(Fs)
            self.ds = np.array(ds)
            self.inv_etas = np.array(inv_etas)


class GaussianCellTypeEmissions(_GaussianEmissionsMixin, _LinearEmissions):
    """ Gaussian emissions with a block diagonal structure for the emission matrix, each block corresponding to a different cell type, we also enforce non-negative weights for the emission matrix when there is more than one cell type """

    def __init__(self, N, K, D, M=0, single_subspace=True, cell_identity=None, region_identity=None, list_of_dimensions=None, **kwargs):
        super(GaussianCellTypeEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        self.cell_identity = cell_identity  # defines the cell class identity of each neuron, 0 for unknown
        # assert that the cell identity is not None and contains non-negative integers
        assert self.cell_identity is not None, "Cell identity must be provided"
        assert np.all(np.equal(np.mod(self.cell_identity, 1), 0)), "cell identity must be integers"
        assert np.all(self.cell_identity >= 0), "cell identity must be non-negative"
        if 0 in self.cell_identity:
            print("Warning: cell identity contains 0, which is reserved for unknown cell types")
        self.region_identity = region_identity  # defines the region identity of each neuron, used to enforce block diagonal structure
        if self.region_identity is not None:
            assert np.all(self.region_identity >= 0), "region identity must be non-negative"
            assert np.all(np.equal(np.mod(self.region_identity, 1), 0)), "cell identity must be integers"
            if np.min(self.region_identity) > 0:
                self.region_identity  = self.region_identity - np.min(self.region_identity)
        else:
            self.region_identity = np.zeros(N, dtype=int)
        self.list_of_dimensions = list_of_dimensions.astype(int)  # defines the dimensions of the latent space for each region

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """ initialize with PCA similar to standard GaussianEmissions """
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = pca.noise_variance_ + 1e-4*np.eye(self.N)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        T, D = data.shape
        if self.single_subspace:
            # we may have neurons whose cell identity is unknown, in which case we remove them from this computation
            if self.cell_identity is not None:
                R_inv = np.linalg.inv(self.inv_etas[0][self.cell_identity!=0, :][:, self.cell_identity!=0])
                block = -1.0 * (self.Cs[0][self.cell_identity!=0,:]).T@R_inv@(self.Cs[0][self.cell_identity!=0, :])
            else:
                R_inv = np.linalg.inv(self.inv_etas[0])
                block = -1.0 * self.Cs[0].T@R_inv@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@inv_eta@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess
    
    def log_prior(self,):
        """ computes the log prior of the emission parameters """
        log_prior = 0
        # we diagonalize noise, hence we can get rid of the prior
        # K = 1 if self.single_subspace else self.K
        # for k in range(K):
        #     # compute inverse wishart prior on the emission noise 
        #     log_prior += log_prior + invwishart.logpdf(self.inv_etas[k], self.nu0[k] + self.N +1, self.Psi0[k]*np.eye(self.N))
        return log_prior
    
    def log_likelihoods(self, data, input, mask, tag, x):
        """ compute LL from the emission model, outputs a (T, K) array of log-likelihoods """
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            """ compute log-ll of data under the emission model """
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        mus = self.forward(x, input, tag).reshape((-1, data.shape[0], self.N))
        Sigmas = self.inv_etas

        if self.cell_identity is not None and np.any(self.cell_identity==0): # remove unknown cell types when computing LL (this is during training when we remove unknown cells during the E step)
            mus = mus[:, :, self.cell_identity!=0]
            Sigmas = Sigmas[:, self.cell_identity!=0, :][:, :, self.cell_identity!=0]

        lls = np.column_stack([stats.multivariate_normal_logpdf(data, mu, Sigma)
                               for mu, Sigma in zip(mus, Sigmas)])
        return lls  

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               **kwargs):
        """ M=step for the GaussianCellTypeEmissions which are no longer in closed form"""

        ys = datas
        Xs = [np.column_stack([x]) for (_, x, _, _), u in
                zip(continuous_expectations, inputs)]
        region_identity, cell_identity = self.region_identity, self.cell_identity

        ExxT, ExyT, EyyT, weight_sum = self._compute_statistics_for_m_step(ys, inputs, masks, tags, continuous_expectations, discrete_expectations)
        ws = [Ez for (Ez, _, _) in discrete_expectations]
        list_of_dims = kwargs.get('list_of_dimensions') # how to partition the latent space dimensionality to create the block diagonal structure
        
        if self.single_subspace and all([np.all(mask) for mask in masks]):
            # Return exact m-step updates for C, F, d, and inv_etas
            initial_C = np.hstack(([self.Cs[0], self.ds[0][:, None]]))
            # get expectations in right shape
            expectations = [ExxT[0], ExyT[0], EyyT[0], weight_sum]
            CF, d, Sigma = fit_constrained_linear_regression(
                Xs, ys, 
                expectations=expectations, 
                Psi0=self.Psi0[0], nu0=self.nu0[0],
                prior_ExxT=1e-4 * np.eye(self.D+1),
                prior_ExyT=np.zeros((self.D+1, self.N)), 
                list_of_dims = list_of_dims, 
                region_identity = region_identity, cell_identity = cell_identity,
                initial_C=initial_C) 
            self.Cs = CF[None, :, :self.D]
            self.inv_etas =  Sigma[None, :]
            self.Fs = np.zeros((1, self.N, self.M))
            self.ds = d[None, :]
        else: 
            # for a switching model, we need to fit a separate emission model for each discrete state
            Cs, Fs, ds, inv_etas = [], [], [], []
            for k in range(self.K):
                initial_C = np.hstack(([self.Cs[k], self.Fs[k], self.ds[k][:, None]]))
                expectations = [ExxT[k], ExyT[k], EyyT[k], weight_sum]
                CF, d, Sigma = fit_constrained_linear_regression(
                    Xs, ys, 
                    expectations = expectations,
                    weights=[w[:, k] for w in ws],
                    Psi0=self.Psi0[k], nu0=self.nu0[k],
                    prior_ExxT=1e-4* np.eye(self.D + self.M + 1),
                    prior_ExyT=np.zeros((self.D + self.M + 1, self.N)),
                    list_of_dims = list_of_dims,
                    region_identity = region_identity,
                    cell_identity = cell_identity,
                    initial_C=initial_C)
                Cs.append(CF[:, :self.D])
                Fs.append(CF[:, self.D:])
                ds.append(d)
            self.Cs = np.array(Cs)
            self.Fs = np.array(Fs)
            self.ds = np.array(ds)
            self.inv_etas = np.array(inv_etas)


class GaussianOrthogonalEmissions(_GaussianEmissionsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
            hess = np.tile(block[None,:,:], (T, 1, 1))
        else:
            blocks = np.array([-1.0 * C.T@np.diag(1.0/np.exp(inv_eta))@C
                               for C, inv_eta in zip(self.Cs, self.inv_etas)])
            hess = np.sum(Ez[:,:,None,None] * blocks, axis=1)
        return -1 * hess


class GaussianIdentityEmissions(_GaussianEmissionsMixin, _IdentityEmissions):

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        if self.single_subspace:
            block = -1.0 * np.diag(1.0 / np.exp(self.inv_etas[0]))
            hess = np.tile(block[None, :, :], (T, 1, 1))
        else:
            raise NotImplementedError

        return -1 * hess


class GaussianNeuralNetworkEmissions(_GaussianEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _StudentsTEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_StudentsTEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones((1, N)) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(_StudentsTEmissionsMixin, self).params + (self.inv_etas, self.inv_nus)

    @params.setter
    def params(self, value):
        super(_StudentsTEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_StudentsTEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        etas, nus = np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.forward(x, input, tag)
        return independent_studentst_logpdf(data[:, None, :],
                                            mus, etas, nus, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        nus = np.exp(self.inv_nus)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[np.arange(T), z, :] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class StudentsTEmissions(_StudentsTEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTOrthogonalEmissions(_StudentsTEmissionsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTIdentityEmissions(_StudentsTEmissionsMixin, _IdentityEmissions):
    pass


class StudentsTNeuralNetworkEmissions(_StudentsTEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _BernoulliEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit", **kwargs):
        super(_BernoulliEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == bool or (data.dtype == int and data.min() >= 0 and data.max() <= 1)
        assert self.link_name == "logit", "Log likelihood is only implemented for logit link."
        logit_ps = self.forward(x, input, tag)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return bernoulli_logpdf(data[:, None, :], logit_ps, mask=mask[:, None, :])

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return (npr.rand(T, self.N) < ps[np.arange(T), z,:]).astype(int)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class BernoulliEmissions(_BernoulliEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess


class BernoulliOrthogonalEmissions(_BernoulliEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        hess = np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])
        return -1 * hess


class BernoulliIdentityEmissions(_BernoulliEmissionsMixin, _IdentityEmissions):
    pass


class BernoulliNeuralNetworkEmissions(_BernoulliEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _PoissonEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0, **kwargs):

        super(_PoissonEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        self.bin_size = bin_size
        mean_functions = dict(
            log=self._log_mean,
            softplus=self._softplus_mean
            )
        self.mean = mean_functions[link]
        link_functions = dict(
            log=self._log_link,
            softplus=self._softplus_link
            )
        self.link = link_functions[link]

        # Set the bias to be small if using log link
        if link == "log":
            self.ds = -3 + .5 * npr.randn(1, N) if single_subspace else npr.randn(K, N)

    def _log_mean(self, x):
        return np.exp(x) * self.bin_size

    def _softplus_mean(self, x):
        return softplus(x) * self.bin_size

    def _log_link(self, rate):
        return np.log(rate) - np.log(self.bin_size)

    def _softplus_link(self, rate):
        return inv_softplus(rate / self.bin_size)

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, np.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[np.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)

class PoissonEmissions(_PoissonEmissionsMixin, _LinearEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(PoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        # Scale down the measurement and control matrices so that
        # rate params don't explode when exponentiated.
        if self.link_name == "log":
            self.Cs /= np.exp(np.linalg.norm(self.Cs, axis=2)[:,:,None])
            self.Fs /= np.exp(np.linalg.norm(self.Fs, axis=2)[:,:,None])

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")

        if self.link_name == "log":
            lambdas = self.mean(self.forward(x, input, tag))
            hess = np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
            return -1 * hess

        elif self.link_name == "softplus":
            # For stability, we avoid evaluating terms that look like exp(x)**2.
            # Instead, we rearrange things so that all terms with exp(x)**2 are of the form
            # (exp(x) / exp(x)**2) which evaluates to sigmoid(x)sigmoid(-x) and avoids overflow.
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0]
            expterms = np.exp(linear_terms)
            outer = logistic(linear_terms) * logistic(-linear_terms)
            diags = outer * (data / lambdas - data / (lambdas**2 * expterms) - self.bin_size)
            hess = np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
            return -hess

        else:
            raise Exception("No Hessian calculation for link: {}".format(self.link_name))


class PoissonOrthogonalEmissions(_PoissonEmissionsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.single_subspace is False:
            raise Exception("Multiple subspaces are not supported for this Emissions class.")

        if self.link_name == "log":
            lambdas = self.mean(self.forward(x, input, tag))
            hess = np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
            return -1 * hess

        elif self.link_name == "softplus":
            # For stability, we avoid evaluating terms that look like exp(x)**2.
            # See comment in PoissoinEmissions.
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0]
            expterms = np.exp(linear_terms)
            outer = logistic(linear_terms) * logistic(-linear_terms)
            diags = outer * (data / lambdas - data / (lambdas**2 * expterms) - self.bin_size)
            hess = np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
            return -hess

        else:
            raise Exception("No Hessian calculation for link: {}".format(self.link_name))

class PoissonIdentityEmissions(_PoissonEmissionsMixin, _IdentityEmissions):
    pass


class PoissonNeuralNetworkEmissions(_PoissonEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _AutoRegressiveEmissionsMixin(object):
    """
    Include past observations as a covariate in the SLDS emissions.
    The AR model is restricted to be diagonal.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_AutoRegressiveEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        # Initialize AR component of the model
        self.As = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Shrink the eigenvalues of the A matrices to avoid instability.
        # Since the As are diagonal, this is just a clip.
        self.As = np.clip(self.As, -1.0 + 1e-8, 1 - 1e-8)

    @property
    def params(self):
        return super(_AutoRegressiveEmissionsMixin, self).params + (self.As, self.inv_etas)

    @params.setter
    def params(self, value):
        self.As, self.inv_etas = value[-2:]
        super(_AutoRegressiveEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_AutoRegressiveEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.As = self.inv_nus[perm]
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        mus = mus + np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))

        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        assert self.single_subspace, "Can only invert with a single emission model"
        pad = np.zeros((1, self.N))
        resid = data - np.concatenate((pad, self.As * data[:-1]))
        return self._invert(resid, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)

        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + np.sqrt(etas[z[0]]) * npr.randn(N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        mus[1:] += self.As[None, :, :] * data[:-1, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)


class AutoRegressiveEmissions(_AutoRegressiveEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

        # Solve a linear regression for the AR coefficients.
        from sklearn.linear_model import LinearRegression
        for n in range(self.N):
            lr = LinearRegression()
            lr.fit(np.concatenate([d[:-1, n] for d in datas])[:,None],
                   np.concatenate([d[1:, n] for d in datas]))
            self.As[:,n] = lr.coef_[0]

        # Compute the residuals of the AR model
        pad = np.zeros((1,self.N))
        mus = [np.concatenate((pad, self.As[0] * d[:-1])) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class AutoRegressiveOrthogonalEmissions(_AutoRegressiveEmissionsMixin, _OrthogonalLinearEmissions):
    pass


class AutoRegressiveIdentityEmissions(_AutoRegressiveEmissionsMixin, _IdentityEmissions):
    pass


class AutoRegressiveNeuralNetworkEmissions(_AutoRegressiveEmissionsMixin, _NeuralNetworkEmissions):
    pass
