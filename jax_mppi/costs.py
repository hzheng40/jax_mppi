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
Example instantaneous and terminal trajectory costs for MPPI in Autonomous Racing

Author: Hongrui Zheng
Last Modified: Oct/11/2022
"""

import jax
import jax.numpy as jnp
import chex

@jax.jit
def ins_costs(all_traj, race_line):
    """
    Instantaneous costs:
        1. Difference to race line speed
        2. Euclidean distance to race line

    Args:
        all_traj (jnp.DeviceArray (K, T, nx)): candidate trajectories
        race_line (jnp.DeviceArray (num_points, nx)): pre-determined race line on the track
        
    
    Returns:
        costs (jnp.DeviceArray (K, T, 1)): instantaneous costs for all trajectories
    """
    pass

@jax.jit
def terminal_costs(all_traj, opp_pose, scan):
    """
    Terminal costs
        1. Collision indicator cost with opponent and track

    Args:
        all_traj (jnp.DeviceArray (K, T, nx)): candidate trajectories
        opp_pose (jnp.DeviceArray (3, )): opponent pose
        scan (jnp.DeviceArray (num_scan, )): current laser scan
        
        center_line (jnp.DeviceArray (num_points, nx)): pre-determined center line of the track
    
    Returns:
        costs (jnp.DeviceArray (K, 1)): terminal costs for all trajectories
    """
    pass