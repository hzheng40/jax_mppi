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
Single-track Drift model from common road in JAX
https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/tree/master/

Author: Hongrui Zheng
Last Modified: Sep/28/2022
"""

import jax
import jax.numpy as jnp
import chex
from functools import partial

@chex.dataclass
class st_dyn_config:
    # gravity
    g: chex.Scalar = 9.81  # [m/s^2]

    # model switching parameters
    v_s: chex.Scalar = 0.2  # switching velocity [m/s]
    v_b: chex.Scalar = 0.05  #
    v_min: chex.Scalar = v_s / 2.0  #

    # vehicle parameters
    length: chex.Scalar = 4.298  # vehicle length [m]
    width: chex.Scalar = 1.674  # vehicle width [m]
    m: chex.Scalar = 1225.887  # vehicle mass [kg]
    I_z: chex.Scalar = 1538.853371  # moment of inertia in yaw []
    lf: chex.Scalar = 0.88392  # distance CoG to front axle [m]
    lr: chex.Scalar = 1.50876  # distance CoG to rear axle [m]
    h_cg: chex.Scalar = 0.557784  # distance CoG above ground [m]
    R_w: chex.Scalar = 0.344  # effective wheel radius []
    I_y_w: chex.Scalar = 1.7  # wheel moment of inertia []
    T_sb: chex.Scalar = 0.76  # split of brake torque []
    T_se: chex.Scalar = 1.0  # split of engine torque []

    # wheel parameters for longitudinal pure slip
    tire_p_cx1: chex.Scalar = 1.6411  # shape factor Cfx for long. force []
    tire_p_dx1: chex.Scalar = 1.1739  # longitudinal friction mu_x at Fznom []
    tire_p_dx3: chex.Scalar = 0.0  # variation of friction mu_x with camber []
    tire_p_ex1: chex.Scalar = 0.46403  # longitudinal curvature Efx at Fznom []
    tire_p_kx1: chex.Scalar = 22.303  # longitudinal slip stiffness Kfx/Fz at Fznom []
    tire_p_hx1: chex.Scalar = 0.0012297  # horizontal shift Shx at Fznom []
    tire_p_vx1: chex.Scalar = -8.8098e-006  # vertical shift Svx/Fz at Fznom []

    # wheel parameters for longitudinal combined slip
    tire_r_bx1: chex.Scalar = 13.276  # slope factor for combined slip Fx reduction []
    tire_r_bx2: chex.Scalar = -13.778  # variation of slop Fx reduction with kappa []
    tire_r_cx1: chex.Scalar = 1.2568  # shape factor for combined slip Fx reduction []
    tire_r_ex1: chex.Scalar = 0.65225  # curvature factor for combined Fx []
    tire_r_hx1: chex.Scalar = 0.0050722  # shift factor for combined slip Fx reduction []

    # wheel parameters for lateral pure slip
    tire_p_cy1: chex.Scalar = 1.3507  # shape factor Cfy for lateral forces []
    tire_p_dy1: chex.Scalar = 1.0489  # lateral friction Muy []
    tire_p_dy3: chex.Scalar = -2.8821  # variation of friction Muy with squared camber []
    tire_p_ey1: chex.Scalar = -0.0074722  # lateral curvature Efy at Fznom []
    tire_p_ky1: chex.Scalar = -21.92  # maximum value of stiffness Kfy/Fznom []
    tire_p_hy1: chex.Scalar = 0.0026747  # horizontal shift Shy at Fznom []
    tire_p_hy3: chex.Scalar = 0.031415  # variation of shift Shy with camber []
    tire_p_vy1: chex.Scalar = 0.037318  # vertical shift in Svy/Fz at Fznom []
    tire_p_vy3: chex.Scalar = -0.32931  # variation of shift Svy/Fz with camber []

    # wheel parameters for lateral combined slip
    tire_r_by1: chex.Scalar = 7.1433  # slope factor for combined Fy reduction []
    tire_r_by2: chex.Scalar = 9.1916  # variation of slope Fy reduction with alpha []
    tire_r_by3: chex.Scalar = -0.027856  # shift term for alpha in slope Fy reduction []
    tire_r_cy1: chex.Scalar = 1.0719  # shape factor for combined Fy reduction []
    tire_r_ey1: chex.Scalar = -0.27572  # curvature factor of combined Fy []
    tire_r_hy1: chex.Scalar = 5.7448e-006  # shift factor for combined Fy reduction []
    tire_r_vy1: chex.Scalar = -0.027825  # Kappa induced side force Svyk/Muy*Fz at Fznom []
    tire_r_vy3: chex.Scalar = -0.27568  # variation of Svyk/Muy*Fz with camber []
    tire_r_vy4: chex.Scalar = 12.12  # variation of Svyk/Muy*Fz with alpha []
    tire_r_vy5: chex.Scalar = 1.9  # variation of Svyk/Muy*Fz with kappa []
    tire_r_vy6: chex.Scalar = -10.704  # variation of Svyk/Muy*Fz with arctan(kappa) []

    # state indices
    X: chex.Scalar = 0
    Y: chex.Scalar = 1
    STEERING_ANGLE: chex.Scalar = 2
    V: chex.Scalar = 3
    YAW: chex.Scalar = 4
    YAW_RATE: chex.Scalar = 5
    SIDE_SLIP: chex.Scalar = 6
    FRONT_WHEEL_SPEED: chex.Scalar = 7
    REAR_WHEEL_SPEED: chex.Scalar = 8

    # control indices
    STEER_SPEED: chex.Scalar = 0
    ACCELERATION: chex.Scalar = 1


@jax.jit
def formula_longitudinal(kappa, gamma, F_z, dyn_config):

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    kappa = -kappa

    S_hx = dyn_config.tire_p_hx1
    S_vx = F_z * dyn_config.tire_p_vx1

    kappa_x = kappa + S_hx
    mu_x = dyn_config.tire_p_dx1 * (1 - dyn_config.tire_p_dx3 * gamma**2)

    C_x = dyn_config.tire_p_cx1
    D_x = mu_x * F_z
    E_x = dyn_config.tire_p_ex1
    K_x = F_z * dyn_config.tire_p_kx1
    B_x = K_x / (C_x * D_x)

    # magic tire formula
    return D_x * jnp.sin(
        C_x
        * jnp.arctan(B_x * kappa_x - E_x * (B_x * kappa_x - jnp.arctan(B_x * kappa_x)))
        + S_vx
    )


# lateral tire forces
@jax.jit
def formula_lateral(alpha, gamma, F_z, dyn_config):

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    # coordinate system transformation
    # alpha = -alpha

    S_hy = jnp.sign(gamma) * (
        dyn_config.tire_p_hy1 + dyn_config.tire_p_hy3 * jnp.abs(gamma)
    )
    S_vy = (
        jnp.sign(gamma)
        * F_z
        * (dyn_config.tire_p_vy1 + dyn_config.tire_p_vy3 * jnp.abs(gamma))
    )

    alpha_y = alpha + S_hy
    mu_y = dyn_config.tire_p_dy1 * (1 - dyn_config.tire_p_dy3 * gamma**2)

    C_y = dyn_config.tire_p_cy1
    D_y = mu_y * F_z
    E_y = dyn_config.tire_p_ey1
    K_y = F_z * dyn_config.tire_p_ky1  # simplify K_y0 to tire_p_ky1*F_z
    B_y = K_y / (C_y * D_y)

    # magic tire formula
    F_y = (
        D_y
        * jnp.sin(
            C_y
            * jnp.arctan(
                B_y * alpha_y - E_y * (B_y * alpha_y - jnp.arctan(B_y * alpha_y))
            )
        )
        + S_vy
    )

    return F_y, mu_y


# longitudinal tire forces for combined slip
@jax.jit
def formula_longitudinal_comb(kappa, alpha, F0_x, dyn_config):

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hxalpha = dyn_config.tire_r_hx1

    alpha_s = alpha + S_hxalpha

    B_xalpha = dyn_config.tire_r_bx1 * jnp.cos(
        jnp.arctan(dyn_config.tire_r_bx2 * kappa)
    )
    C_xalpha = dyn_config.tire_r_cx1
    E_xalpha = dyn_config.tire_r_ex1
    D_xalpha = F0_x / (
        jnp.cos(
            C_xalpha
            * jnp.arctan(
                B_xalpha * S_hxalpha
                - E_xalpha * (B_xalpha * S_hxalpha - jnp.arctan(B_xalpha * S_hxalpha))
            )
        )
    )

    # magic tire formula
    return D_xalpha * jnp.cos(
        C_xalpha
        * jnp.arctan(
            B_xalpha * alpha_s
            - E_xalpha * (B_xalpha * alpha_s - jnp.arctan(B_xalpha * alpha_s))
        )
    )


# lateral tire forces for combined slip
@jax.jit
def formula_lateral_comb(kappa, alpha, gamma, mu_y, F_z, F0_y, dyn_config):

    # turn slip is neglected, so xi_i=1
    # all scaling factors lambda = 1

    S_hykappa = dyn_config.tire_r_hy1

    kappa_s = kappa + S_hykappa

    B_ykappa = dyn_config.tire_r_by1 * jnp.cos(
        jnp.arctan(dyn_config.tire_r_by2 * (alpha - dyn_config.tire_r_by3))
    )
    C_ykappa = dyn_config.tire_r_cy1
    E_ykappa = dyn_config.tire_r_ey1
    D_ykappa = F0_y / (
        jnp.cos(
            C_ykappa
            * jnp.arctan(
                B_ykappa * S_hykappa
                - E_ykappa * (B_ykappa * S_hykappa - jnp.arctan(B_ykappa * S_hykappa))
            )
        )
    )

    D_vykappa = (
        mu_y
        * F_z
        * (dyn_config.tire_r_vy1 + dyn_config.tire_r_vy3 * gamma)
        * jnp.cos(jnp.arctan(dyn_config.tire_r_vy4 * alpha))
    )
    S_vykappa = D_vykappa * jnp.sin(
        dyn_config.tire_r_vy5 * jnp.arctan(dyn_config.tire_r_vy6 * kappa)
    )

    # magic tire formula
    return (
        D_ykappa
        * jnp.cos(
            C_ykappa
            * jnp.arctan(
                B_ykappa * kappa_s
                - E_ykappa * (B_ykappa * kappa_s - jnp.arctan(B_ykappa * kappa_s))
            )
        )
        + S_vykappa
    )


@jax.jit
def update_dyn_kinematic(states, control_inputs, dyn_config):
    """
    Get the evaluated ODEs of the state at this point

    Args:
        states (): Shape of (B, 5) or (5)
        control_inputs (): Shape of (B, 2) or (2)
    """
    diff = jnp.zeros_like(states)

    l_wb = dyn_config.lf + dyn_config.lr

    # slip angle (beta) from vehicle kinematics
    beta = jnp.arctan(jnp.tan(states[dyn_config.STEERING_ANGLE]) * dyn_config.lr / l_wb)

    diff = diff.at[dyn_config.X].set(
        states[dyn_config.V] * jnp.cos(beta + states[dyn_config.YAW])
    )
    diff = diff.at[dyn_config.Y].set(
        states[dyn_config.V] * jnp.sin(beta + states[dyn_config.YAW])
    )
    diff = diff.at[dyn_config.STEERING_ANGLE].set(
        control_inputs[dyn_config.STEER_SPEED]
    )
    diff = diff.at[dyn_config.V].set(control_inputs[dyn_config.ACCELERATION])
    diff = diff.at[dyn_config.YAW].set(
        (
            states[dyn_config.V]
            * jnp.cos(beta)
            * jnp.tan(states[dyn_config.STEERING_ANGLE])
            / l_wb
        )
    )

    return diff


@jax.jit
def update_dyn_std(states, control_inputs, dyn_config):
    """
    Get the evaluated ODEs of the state at this point

    Args:
        states (): Shape of (B, 9) or (9)
        control_inputs (): Shape of (B, 2) or (2)
    """
    diff = jnp.zeros_like(states)

    lwb = dyn_config.lf + dyn_config.lr

    # compute lateral tire slip angles
    # JAX cond flow
    alpha_f = jax.lax.cond(
        states[dyn_config.V] > dyn_config.v_min,
        alphaf,
        alphazero,
        states[dyn_config.V],
        states[dyn_config.SIDE_SLIP],
        states[dyn_config.YAW_RATE],
        dyn_config.lf,
        states[dyn_config.STEERING_ANGLE],
    )
    alpha_r = jax.lax.cond(
        states[dyn_config.V] > dyn_config.v_min,
        alphar,
        alphazero,
        states[dyn_config.V],
        states[dyn_config.SIDE_SLIP],
        states[dyn_config.YAW_RATE],
        dyn_config.lr,
        states[dyn_config.STEERING_ANGLE],
    )

    # non-JAX if-else
    # if states[dyn_config.V] > dyn_config.v_min:
    #     alpha_f = jnp.arctan(
    #         (
    #             states[dyn_config.V] * jnp.sin(states[dyn_config.SIDE_SLIP])
    #             + states[dyn_config.YAW_RATE] * dyn_config.lf
    #         )
    #         / (states[dyn_config.V] * jnp.cos(states[dyn_config.SIDE_SLIP]))
    #     )
    #     -states[dyn_config.STEERING_ANGLE]

    #     alpha_r = jnp.arctan(
    #         (
    #             states[dyn_config.V] * jnp.sin(states[dyn_config.SIDE_SLIP])
    #             - states[dyn_config.YAW_RATE] * dyn_config.lr
    #         )
    #         / (states[dyn_config.V] * jnp.cos(states[dyn_config.SIDE_SLIP]))
    #     )
    # else:
    #     alpha_f = 0.0
    #     alpha_r = 0.0

    # compute vertical tire forces
    F_zf = (
        dyn_config.m
        * (
            -control_inputs[dyn_config.ACCELERATION] * dyn_config.h_cg
            + dyn_config.g * dyn_config.lr
        )
        / (dyn_config.lr + dyn_config.lf)
    )
    F_zr = (
        dyn_config.m
        * (
            control_inputs[dyn_config.ACCELERATION] * dyn_config.h_cg
            + dyn_config.g * dyn_config.lf
        )
        / (dyn_config.lr + dyn_config.lf)
    )

    # compute front and rear tire speeds, speed of tires can be only positive
    u_wf = jnp.maximum(
        0.0,
        states[dyn_config.V]
        * jnp.cos(states[dyn_config.SIDE_SLIP])
        * jnp.cos(states[dyn_config.STEERING_ANGLE])
        + (
            states[dyn_config.V] * jnp.sin(states[dyn_config.SIDE_SLIP])
            + dyn_config.lf * states[dyn_config.YAW_RATE]
        )
        * jnp.sin(states[dyn_config.STEERING_ANGLE]),
    )
    u_wr = jnp.maximum(
        0.0,
        states[dyn_config.V] * jnp.cos(states[dyn_config.SIDE_SLIP]),
    )

    # compute longitudinal tire slip
    s_f = 1 - dyn_config.R_w * states[dyn_config.FRONT_WHEEL_SPEED] / jnp.maximum(
        u_wf, dyn_config.v_min
    )
    s_r = 1 - dyn_config.R_w * states[dyn_config.REAR_WHEEL_SPEED] / jnp.maximum(
        u_wr, dyn_config.v_min
    )

    # compute tire forces (Pacejka)
    # pure slip longitudinal forces
    F0_xf = formula_longitudinal(s_f, 0, F_zf, dyn_config)
    F0_xr = formula_longitudinal(s_r, 0, F_zr, dyn_config)

    # pure slip lateral forces
    F0_yf, mu_yf = formula_lateral(alpha_f, 0, F_zf, dyn_config)
    F0_yr, mu_yr = formula_lateral(alpha_r, 0, F_zr, dyn_config)

    # combined slip longitudinal forces
    F_xf = formula_longitudinal_comb(s_f, alpha_f, F0_xf, dyn_config)
    F_xr = formula_longitudinal_comb(s_r, alpha_r, F0_xr, dyn_config)

    # combined slip lateral forces
    F_yf = formula_lateral_comb(s_f, alpha_f, 0, mu_yf, F_zf, F0_yf, dyn_config)
    F_yr = formula_lateral_comb(s_r, alpha_r, 0, mu_yr, F_zr, F0_yr, dyn_config)

    # convert acceleration input to brake and engine torque
    # JAX cond flow
    T_B = jax.lax.cond(
        control_inputs[dyn_config.ACCELERATION] > 0.0,
        tzero,
        t,
        dyn_config.m,
        dyn_config.R_w,
        control_inputs[dyn_config.ACCELERATION],
    )
    T_E = jax.lax.cond(
        control_inputs[dyn_config.ACCELERATION] > 0.0,
        t,
        tzero,
        dyn_config.m,
        dyn_config.R_w,
        control_inputs[dyn_config.ACCELERATION],
    )

    # non-JAX if-else
    # if control_inputs[dyn_config.ACCELERATION] > 0:
    #     T_B = 0.0
    #     T_E = dyn_config.m * dyn_config.R_w * control_inputs[dyn_config.ACCELERATION]
    # else:
    #     T_B = dyn_config.m * dyn_config.R_w * control_inputs[dyn_config.ACCELERATION]
    #     T_E = 0.0

    # system dynamics
    d_v = (
        1
        / dyn_config.m
        * (
            -F_yf
            * jnp.sin(states[dyn_config.STEERING_ANGLE] - states[dyn_config.SIDE_SLIP])
            + F_yr * jnp.sin(states[dyn_config.SIDE_SLIP])
            + F_xr * jnp.cos(states[dyn_config.SIDE_SLIP])
            + F_xf
            * jnp.cos(states[dyn_config.STEERING_ANGLE] - states[dyn_config.SIDE_SLIP])
        )
    )
    dd_psi = (
        1
        / dyn_config.I_z
        * (
            F_yf * jnp.cos(states[dyn_config.STEERING_ANGLE]) * dyn_config.lf
            - F_yr * dyn_config.lr
            + F_xf * jnp.sin(states[dyn_config.STEERING_ANGLE]) * dyn_config.lf
        )
    )

    # non-JAX if-else
    # if states[dyn_config.V] > dyn_config.v_min:
    #     d_beta = -states[dyn_config.YAW_RATE] + 1 / (
    #         dyn_config.m * states[dyn_config.V]
    #     ) * (
    #         F_yf
    #         * jnp.cos(states[dyn_config.STEERING_ANGLE] - states[dyn_config.SIDE_SLIP])
    #         + F_yr * jnp.cos(states[dyn_config.SIDE_SLIP])
    #         - F_xr * jnp.sin(states[dyn_config.SIDE_SLIP])
    #         + F_xf
    #         * jnp.sin(states[dyn_config.STEERING_ANGLE] - states[dyn_config.SIDE_SLIP])
    #     )
    # else:
    #     d_beta = 0.0

    # JAX control flow cond
    d_beta = jax.lax.cond(
        states[dyn_config.V] > dyn_config.v_min,
        dbeta,
        dbetazero,
        states[dyn_config.YAW_RATE],
        dyn_config.m,
        states[dyn_config.V],
        F_yf,
        states[dyn_config.STEERING_ANGLE],
        states[dyn_config.SIDE_SLIP],
        F_yr,
        F_xr,
        F_xf,
    )


    # wheel dynamics (negative wheel spin forbidden)

    # Non-JAX if-else
    # if states[dyn_config.FRONT_WHEEL_SPEED] >= 0:
    #     d_omega_f = (
    #         1.0
    #         / dyn_config.I_y_w
    #         * (-dyn_config.R_w * F_xf + dyn_config.T_sb * T_B + dyn_config.T_se * T_E)
    #     )
    # else:
    #     d_omega_f = 0.0
    
    # states = states.at[dyn_config.FRONT_WHEEL_SPEED].max(0.0)

    # if states[dyn_config.REAR_WHEEL_SPEED] >= 0:
    #     d_omega_r = (
    #         1.0
    #         / dyn_config.I_y_w
    #         * (
    #             -dyn_config.R_w * F_xr
    #             + (1.0 - dyn_config.T_sb) * T_B
    #             + (1.0 - dyn_config.T_se) * T_E
    #         )
    #     )
    # else:
    #     d_omega_r = 0.0
    
    # states = states.at[dyn_config.REAR_WHEEL_SPEED].max(0.0)

    # JAX control flow cond
    d_omega_f = jax.lax.cond(
        states[dyn_config.FRONT_WHEEL_SPEED] >= 0.0,
        df,
        dfzero,
        dyn_config.I_y_w,
        dyn_config.R_w,
        F_xf,
        dyn_config.T_sb,
        T_B,
        dyn_config.T_se,
        T_E,
    )
    states = states.at[dyn_config.FRONT_WHEEL_SPEED].max(0.0)
    
    d_omega_r = jax.lax.cond(
        states[dyn_config.REAR_WHEEL_SPEED] >= 0.0,
        dr,
        drzero,
        dyn_config.I_y_w,
        dyn_config.R_w,
        F_xr,
        dyn_config.T_sb,
        T_B,
        dyn_config.T_se,
        T_E,
    )
    states = states.at[dyn_config.REAR_WHEEL_SPEED].max(0.0)
    

    # *** Mix with kinematic model at low speeds ***
    # kinematic system dynamics
    f_ks = update_dyn_kinematic(states, control_inputs, dyn_config)
    # derivative of slip angle and yaw rate (kinematic)
    d_beta_ks = (dyn_config.lr * control_inputs[dyn_config.STEER_SPEED]) / (
        lwb
        * jnp.cos(states[dyn_config.STEERING_ANGLE]) ** 2
        * (
            1
            + (jnp.tan(states[dyn_config.STEERING_ANGLE]) ** 2 * dyn_config.lr / lwb)
            ** 2
        )
    )
    dd_psi_ks = (
        1
        / lwb
        * (
            control_inputs[dyn_config.ACCELERATION]
            * jnp.cos(states[dyn_config.SIDE_SLIP])
            * jnp.tan(states[dyn_config.STEERING_ANGLE])
            - states[dyn_config.V]
            * jnp.sin(states[dyn_config.SIDE_SLIP])
            * d_beta_ks
            * jnp.tan(states[dyn_config.STEERING_ANGLE])
            + states[dyn_config.V]
            * jnp.cos(states[dyn_config.SIDE_SLIP])
            * control_inputs[dyn_config.STEER_SPEED]
            / jnp.cos(states[dyn_config.STEERING_ANGLE]) ** 2
        )
    )
    # derivative of angular speeds (kinematic)
    d_omega_f_ks = (1 / 0.02) * (
        u_wf / dyn_config.R_w - states[dyn_config.FRONT_WHEEL_SPEED]
    )
    d_omega_r_ks = (1 / 0.02) * (
        u_wr / dyn_config.R_w - states[dyn_config.REAR_WHEEL_SPEED]
    )

    # weights for mixing both models
    w_std = 0.5 * (
        jnp.tanh((states[dyn_config.V] - dyn_config.v_s) / dyn_config.v_b) + 1
    )
    w_ks = 1 - w_std

    # output vector: mix results of dynamic and kinematic model
    diff = diff.at[dyn_config.X].set(
        states[dyn_config.V]
        * jnp.cos(states[dyn_config.SIDE_SLIP] + states[dyn_config.YAW])
    )
    diff = diff.at[dyn_config.Y].set(
        states[dyn_config.V]
        * jnp.sin(states[dyn_config.SIDE_SLIP] + states[dyn_config.YAW])
    )
    diff = diff.at[dyn_config.STEERING_ANGLE].set(
        control_inputs[dyn_config.STEER_SPEED]
    )
    diff = diff.at[dyn_config.V].set(w_std * d_v + w_ks * f_ks[dyn_config.V])
    diff = diff.at[dyn_config.YAW].set(
        (w_std * states[dyn_config.YAW_RATE] + w_ks * f_ks[dyn_config.YAW])
    )
    diff = diff.at[dyn_config.YAW_RATE].set(w_std * dd_psi + w_ks * dd_psi_ks)
    diff = diff.at[dyn_config.SIDE_SLIP].set(w_std * d_beta + w_ks * d_beta_ks)
    diff = diff.at[dyn_config.FRONT_WHEEL_SPEED].set(
        w_std * d_omega_f + w_ks * d_omega_f_ks
    )
    diff = diff.at[dyn_config.REAR_WHEEL_SPEED].set(
        w_std * d_omega_r + w_ks * d_omega_r_ks
    )

    return diff


@jax.jit
def tzero(a, b, c):
    return 0.0


@jax.jit
def t(a, b, c):
    return a * b * c


@jax.jit
def df(a, b, c, d, e, f, g):
    return 1.0 / a * (-b * c + d * e + f * g)


@jax.jit
def dfzero(a, b, c, d, e, f, g):
    return 0.0


@jax.jit
def dr(a, b, c, d, e, f, g):
    return 1.0 / a * (-b * c + (1.0 - d) * e + (1.0 - f) * g)


@jax.jit
def drzero(a, b, c, d, e, f, g):
    return 0.0


@jax.jit
def alphaf(a, b, c, d, e):
    return jnp.arctan((a * jnp.sin(b) + c * d) / (a * jnp.cos(b))) - e


@jax.jit
def alphar(a, b, c, d, e):
    return jnp.arctan((a * jnp.sin(b) - c * d) / (a * jnp.cos(b)))


@jax.jit
def alphazero(a, b, c, d, e):
    return 0.0


@jax.jit
def dbeta(a, b, c, d, e, f, g, h, i):
    return -a + 1.0 / (b * c) * (
        d * jnp.cos(e - f) + g * jnp.cos(f) - h * jnp.sin(f) + i * jnp.sin(e - f)
    )


@jax.jit
def dbetazero(a, b, c, d, e, f, g, h, i):
    return 0.0



def rollout(dyn_fun, state_init, u, N):
    """
    state_init (jnp array (B, 9)): initial states
    u (jnp array (B, 2)): 
    """
    
    last_state, all_states = jax.lax.scan(jax.vmap(dyn_fun, in_axes=(0, 0)), state_init, u, length=N)

    return last_state, all_states

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import timeit
    
    batch_size = 2048
    dT = 0.1
    N = 100

    # TODO: use jax.lax.scan instead of for loop
    key = jax.random.PRNGKey(123)
    u_mean = jnp.zeros((batch_size, 2))
    u_seq = jax.random.multivariate_normal(key, jnp.zeros((batch_size, 2)))

    conf = st_dyn_config()
    update_dyn_std_partial = partial(update_dyn_std, dyn_config=conf)

    
    states = jnp.zeros((batch_size, 9))
    states = states.at[:, 2].add(0.1)
    
    all_states = []
    u = jnp.zeros((batch_size, 2))
    u = u.at[:, 1].set(jnp.linspace(5.0, 10.0, batch_size))

    jax.vmap(update_dyn_std, in_axes=(0, 0, None))(states, u, conf)

    for i in range(N):
        all_states.append(states)
        states = states + jax.vmap(update_dyn_std, in_axes=(0, 0, None))(states, u, conf) * dT
    
    all_states = jnp.array(all_states)
    
    def wrap_vmap_cpu():
        jax.vmap(jax.jit(update_dyn_std, backend='cpu'), in_axes=(0, 0, None))(states, u, conf)
    
    def wrap_cpu():
        jax.jit(update_dyn_std, backend='cpu')(states[0, :], u[:, 0], conf)
    
    def wrap_vmap_gpu():
        jax.vmap(jax.jit(update_dyn_std, backend='gpu'), in_axes=(0, 0, None))(states, u, conf)
    
    def wrap_gpu():
        jax.jit(update_dyn_std, backend='gpu')(states[0, :], u[:, 0], conf)

    print('CPU backend runtime 1000, single batch: ', timeit.timeit(wrap_cpu, number=1000))
    print('CPU backend runtime 1000: 2048 batch', timeit.timeit(wrap_vmap_cpu, number=1000))
    print('GPU backend runtime 1000: single batch', timeit.timeit(wrap_gpu, number=1000))
    print('GPU backend runtime 1000: 2048 batch', timeit.timeit(wrap_vmap_gpu, number=1000))

    # plt.plot(all_states[:, :, 0], all_states[:, :, 1])
    # plt.axis("equal")
    # plt.show()

    # states_label = ["x", "y", "delta", "vel", "yaw", "yaw rate", "beta", "fwv", "rwv"]
    # fig, ax_list = plt.subplots(nrows=9, ncols=1)
    # for i, ax in enumerate(ax_list):
    #     ax.plot(all_states[:, :, i])
    #     ax.set_ylabel(states_label[i])

    # plt.show()
