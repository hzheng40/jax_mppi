# MIT License

# Copyright (c) 2022 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Model Predictive Path Integral Control with JAX
Implementation follows Algorithm 2 in
Grady et al. "Information Theoretic MPC for Model-Based Reinforcement Learning"
Author: Hongrui Zheng
Last Modified: Oct/10/2022
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from st_drift import st_dyn_config, update_dyn_std

# disable vram preallocate
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class MPPI:
    def __init__(
        self,
        dyn_fun,
        nx,
        nu,
        noise_sigma,
        terminal_state_cost,
        ins_state_cost,
        num_samples=2048,
        horizon=100,
        lamb=1.0,
        seed=69420,
        noise_mu=None,
        u_min=None,
        u_max=None,
        u_end_init=None,
        U_init=None,
    ):
        """
        Model Predictive Path Integral Control in JAX

        Args:
            dyn_fun (function(state (jnp.DeviceArray, (K, nx)),
                              input (jnp.DeviceArray, (K, nu)))
                              -> next_state (jnp.DeviceArray, (K, nx)))
                    dynamics of the system
            nx (int): size of state vector
            nu (int): size of input vector
            noise_sigma (jnp.DeviceArray, (nu, nu)): covariance of input noise, Sigma in alg2
            terminal_state_cost (function (state (jnp.DeviceArray, (K, nx))
                                           -> cost (jnp.DeviceArray, (K, 1))))
                                costs on terminal state of trajectories, phi in alg2
            ins_state_cost (function (trajectories (jnp.DeviceArray, (K, T, nx))
                                      -> costs (jnp.DeviceArray, (K, T, 1))))
                            costs on instantaneous states, q in alg2

            num_samples (int, default=2048): number of trajectories to sample
            horizon (int, default=100): horizon of each trajectory
            lamb (float, default=1.0): temperature, larger encourages more exploration, lambda in alg2
            seed (int): seed for jax.random
            noise_mu (jnp.DeviceArray, (nu, ), default=None): mean of noise, biases noise
            u_min (jnp.DeviceArray (nu, ), default=None): minimum values for control inputs
            u_max (jnp.DeviceArray (nu, ), default=None): maximum values for control inputs
            u_end_init (jnp.DeviceArray (nu, ), default=None): initial value for end of trajectory control
            U_init (jnp.DeviceArray (T, nu), default=None): initial control sequence, default is noise
        """
        # TODO: init dynamics, num sample, horizon, sigma init
        # seeding
        self.rng_key = jrandom.PRNGKey(seed)

        self.F = dyn_fun
        self.K = num_samples
        self.T = horizon
        self.nx = nx
        self.nu = nu
        self.lamb = lamb
        self.Sigma = noise_sigma
        self.phi = terminal_state_cost
        self.q = ins_state_cost

        # noise distribution
        if noise_mu is None:
            noise_mu = jnp.zeros(self.nu)
        self.noise_mu = noise_mu
        self.Sigma_inv = jnp.linalg.inv(self.Sigma)

        if u_end_init is None:
            u_end_init = jnp.zeros_like(noise_mu)

        # initial control sequence
        self.U = U_init
        if self.U is None:
            self.U = jrandom.multivariate_normal(
                self.rng_key, self.noise_mu, self.Sigma, shape=(self.T, self.nu)
            )

        self.u_min = u_min
        self.u_max = u_max

        self.state = None

        # keeping track of previous results
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    def plan(self, state):
        # TODO: sample control inputs based on total cost dist
        # TODO: rollout dynamics
        pass

    def _cal_costs(self):
        # TODO: take rollout trajectories and calculate cost
        # should be jitted and lax.scan wrapped
        # TODO: update self total cost
        pass
