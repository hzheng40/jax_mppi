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
Author: Hongrui Zheng
Last Modified: Sep/28/2022
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from st_drift import st_dyn_config, update_dyn_std


class MPPI:
    def __init__(
        self,
        dyn_fun,
        running_cost,
        nx,
        noise_sigma,
        num_samples=2048,
        horizon=100,
        terminal_state_cost=None,
        lamb=1.0,
        noise_mu=None,
        u_min=None,
        u_max=None,
        U_init=None,
        u_scale=1,
    ) -> None:
        # TODO: init dynamics, num sample, horizon, sigma init
        pass

    def plan(self, state):
        # TODO: sample control inputs based on total cost dist
        # TODO: rollout dynamics
        pass

    def _cal_costs(self):
        # TODO: take rollout trajectories and calculate cost
        # should be jitted and lax.scan wrapped
        # TODO: update self total cost
        pass
