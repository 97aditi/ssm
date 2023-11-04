import copy
import warnings
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad, grad

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, \
    convex_combination, newtons_method_block_tridiag_hessian
from ssm.primitives import hmm_normalizer
from ssm.messages import hmm_expected_states, viterbi, kalman_filter
from ssm.util import ensure_args_are_lists, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, ssm_pbar

import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.emissions as emssn
import ssm.hmm as hmm
import ssm.variational as varinf

class LDS():
    """
   Linear Dynamical System
    """
    def __init__(self, N, D, *, M=0,
            dynamics="gaussian",
            dynamics_kwargs=None,
            emissions="gaussian_orthog",
            emission_kwargs=None,
            **kwargs):


        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            gaussian=obs.AutoRegressiveObservations,
            diagonal_gaussian=obs.AutoRegressiveDiagonalNoiseObservations,
            t=obs.RobustAutoRegressiveObservations,
            studentst=obs.RobustAutoRegressiveObservations,
            diagonal_t=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_studentst=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](1, D, M=M, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        # Make the emission distn
        emission_classes = dict(
            gaussian=emssn.GaussianEmissions,
            gaussian_orthog=emssn.GaussianOrthogonalEmissions,
            gaussian_id=emssn.GaussianIdentityEmissions,
            gaussian_nn=emssn.GaussianNeuralNetworkEmissions,
            studentst=emssn.StudentsTEmissions,
            studentst_orthog=emssn.StudentsTOrthogonalEmissions,
            studentst_id=emssn.StudentsTIdentityEmissions,
            studentst_nn=emssn.StudentsTNeuralNetworkEmissions,
            t=emssn.StudentsTEmissions,
            t_orthog=emssn.StudentsTOrthogonalEmissions,
            t_id=emssn.StudentsTIdentityEmissions,
            t_nn=emssn.StudentsTNeuralNetworkEmissions,
            poisson=emssn.PoissonEmissions,
            poisson_orthog=emssn.PoissonOrthogonalEmissions,
            poisson_id=emssn.PoissonIdentityEmissions,
            poisson_nn=emssn.PoissonNeuralNetworkEmissions,
            bernoulli=emssn.BernoulliEmissions,
            bernoulli_orthog=emssn.BernoulliOrthogonalEmissions,
            bernoulli_id=emssn.BernoulliIdentityEmissions,
            bernoulli_nn=emssn.BernoulliNeuralNetworkEmissions,
            ar=emssn.AutoRegressiveEmissions,
            ar_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            ar_id=emssn.AutoRegressiveIdentityEmissions,
            ar_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
            autoregressive=emssn.AutoRegressiveEmissions,
            autoregressive_orthog=emssn.AutoRegressiveOrthogonalEmissions,
            autoregressive_id=emssn.AutoRegressiveIdentityEmissions,
            autoregressive_nn=emssn.AutoRegressiveNeuralNetworkEmissions
            )

        if isinstance(emissions, str):
            emissions = emissions.lower()
            if emissions not in emission_classes:
                raise Exception("Invalid emission model: {}. Must be one of {}".
                    format(emissions, list(emission_classes.keys())))

            emission_kwargs = emission_kwargs or {}
            emissions = emission_classes[emissions](N, 1, D, M=M,
                single_subspace=True, **emission_kwargs)
        if not isinstance(emissions, emssn.Emissions):
            raise TypeError("'emissions' must be a subclass of"
                            " ssm.emissions.Emissions")

        init_state_distn = isd.InitialStateDistribution(1, D, M)
        transitions = trans.StationaryTransitions(1, D, M)
        super().__init__(N, 1, D, M=M,
                         init_state_distn=init_state_distn,
                         transitions=transitions,
                         dynamics=dynamics,
                         emissions=emissions)

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params, \
               self.emissions.params

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        return np.ones((variational_mean.shape[0], 1)), \
               np.ones((variational_mean.shape[0], 1, 1)), \
               0

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def log_prior(self,):
        # TODO: transitions log_prior
        return self.dynamics.log_prior() + self.emissions.log_prior()

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        # let's compute the log likelihood of the data using the kalman filter
        # and then add the log prior
        ll = self.log_likelihood(datas, inputs=inputs, masks=masks, tags=tags)
        return ll + self.log_prior()

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
        # let's compute the log likelihood of the data 

        # get the current model parameters
        As, bs, Vs, _ = self.dynamics.params
        Cs, Fs, ds, inv_etas = self.emissions.params

        # obtain covariances and their inverses
        Q = self.dynamics.Sigmas[0]
        R = inv_etas[0]

        mu0 = self.dynamics.mus_init[0]
        S0 = self.dynamics.Sigmas_init[0]

        ll = 0
        # TODO: not accounting for dynamic bias bs
        for data, input in zip(datas, inputs):
            # accounting for observation bias by subtracting ds[0]
            ll_this, _, _ = kalman_filter(mu0, S0, As[0], Vs[0], Q, Cs[0], Fs[0], R, input, data-ds[0])
            ll += ll_this
        
        return ll 

    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        (_, x, y) = super().sample(T, input=input, tag=tag, prefix=prefix, with_noise=with_noise)
        return (x, y)
