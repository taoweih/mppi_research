import gymnasium as gym
from torch import cos, pi, sin
import torch
import logging
import warnings
import math
from pytorch_mppi import mppi
from gym import logger as gym_log
import time
import custom_envs
import sys
sys.path.append("..")
import custom_mppi
import base_mppi

# gym_log.set_level(gym_log.INFO)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                     datefmt='%m-%d %H:%M:%S')

# params for links
LINK_LENGTH_1 = 1.0  # [m]
LINK_LENGTH_2 = 1.0  # [m]
LINK_MASS_1 = 1.0  #: [kg] mass of link 1
LINK_MASS_2 = 1.0  #: [kg] mass of link 2
LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
LINK_MOI = 1.0  #: moments of inertia for both links

MAX_VEL_1 = 4 * pi
MAX_VEL_2 = 9 * pi

AVAIL_TORQUE = [-1.0, 0.0, +1]

#: use dynamics equations from the nips paper or the book
book_or_nips = "book"

dt_param = 0.2


if __name__ == "__main__":
    ENV_NAME = "ContinuousAcrobot-v1"
    TIMESTEPS = 20  # T
    N_SAMPLES = 20000  # K
    ACTION_LOW = -10.0
    ACTION_HIGH = 10.0

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == torch.device("cpu"):
        d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if d == torch.device("cpu"):
        warnings.warn("No GPU device detected, using cpu instead", UserWarning)

    dtype = torch.float32

    noise_sigma = torch.tensor(3, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.
    
    def wrap(x, m, M):
        """Wraps `x` so m <= x <= M; but unlike `bound()` which
        truncates, `wrap()` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

        Args:
            x: a scalar tensor
            m: minimum possible value in range
            M: maximum possible value in range

        Returns:
            x: a scalar tensor, wrapped
        """
        span = M - m
        return ((x - m) % span) + m

    def bound(x, m, M=None):
        """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

        Args:
            x: scalar tensor
            m: The lower bound (or a 2‑element list/tuple)
            M: The upper bound if m is scalar

        Returns:
            x: scalar tensor, bound between min (m) and Max (M)
        """
        if M is None:
            low, high = m[0], m[1]
        else:
            low, high = m, M
        return torch.clamp(x, low, high)

    def rk4(derivs, y0, t):
        """
        Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

        Example for 2D system:

            >>> def derivs(x):
            ...     d1 =  x[0] + 2*x[1]
            ...     d2 =  -3*x[0] + 4*x[1]
            ...     return d1, d2

            >>> dt = 0.0005
            >>> t = np.arange(0.0, 2.0, dt)
            >>> y0 = (1,2)
            >>> yout = rk4(derivs, y0, t)

        Args:
            derivs: the derivative of the system and has the signature `dy = derivs(yi)`
            y0: initial state vector (torch tensor)
            t: sample times (list or tensor) of length 2: [0, dt]

        Returns:
            yout: Runge-Kutta approximation of the ODE at t[-1], dropping any appended action
        """
        # assume t = [0, dt]
        dt = t[1] - t[0]
        k1 = derivs(y0)
        k2 = derivs(y0 + 0.5 * dt * k1)
        k3 = derivs(y0 + 0.5 * dt * k2)
        k4 = derivs(y0 +       dt * k3)
        y1 = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # return only the state dims (first 4)
        return y1[..., :4]

    def _dsdt(s_augmented):
        m1 = LINK_MASS_1
        m2 = LINK_MASS_2
        l1 = LINK_LENGTH_1
        lc1 = LINK_COM_POS_1
        lc2 = LINK_COM_POS_2
        I1 = LINK_MOI
        I2 = LINK_MOI
        g = 9.8
        a = s_augmented[..., -1]
        s = s_augmented[..., :-1]
        theta1 = s[..., 0]
        theta2 = s[..., 1]
        dtheta1 = s[..., 2]
        dtheta2 = s[..., 3]

        d11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos(theta2)) + I1 + I2
        d12 = m2*(lc2**2 + l1*lc2*cos(theta2)) + I2

        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - math.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - math.pi / 2)
            + phi2
        )

        if book_or_nips == "nips":
            ddtheta2 = (a + d12 / d11 * phi1 - phi2) / (m2 * lc2**2 + I2 - d12**2 / d11)
        else:
            ddtheta2 = (
                a + d12 / d11 * phi1
                - m2 * l1 * lc2 * dtheta1**2 * sin(theta2)
                - phi2
            ) / (m2 * lc2**2 + I2 - d12**2 / d11)

        ddtheta1 = -(d12 * ddtheta2 + phi1) / d11
        zeros = torch.zeros_like(a)
        return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2, zeros], dim=-1)

    def dynamics(state, perturbed_action):
        # true dynamics from gym
        torque = perturbed_action
        s_augmented = torch.cat((state, torque), dim=1)

        ns = rk4(_dsdt, s_augmented, [0, dt_param])
        ns0 = wrap(ns[:, 0], -math.pi, math.pi)
        ns1 = wrap(ns[:, 1], -math.pi, math.pi)
        ns2 = bound(ns[:, 2], -MAX_VEL_1, MAX_VEL_1)
        ns3 = bound(ns[:, 3], -MAX_VEL_2, MAX_VEL_2)

        state = torch.stack([ns0, ns1, ns2, ns3], dim=1)
        return state
    
    def running_cost(state, action):
        theta1 = state[...,0]
        theta2 = state[...,1]
        dtheta1 = state[...,2]
        dtheta2 = state[...,3]
        def wrap_error(x):
            return ((x + math.pi) % (2*math.pi)) - math.pi

        e1 = wrap_error(theta1 - math.pi) 
        e2 = wrap_error(theta2 - 0) 
        cost  = e1*e1 + e2*e2
        cost +=  0.1 * dtheta1 ** 2 + 0.1 * dtheta2 ** 2 
        # cost += 0.01 * action[...,0] * action[...,0]
        return cost


    def train(new_data):
        pass


    downward_start = True
    env = gym.make(ENV_NAME, render_mode="human")
    nx = 4

    # for _ in range(1000):
    #     action = np.array([-2])
    #     _,r,_,_,_ = env.step(action)

    env.reset()
    if downward_start:
        env.state = env.unwrapped.state = [0, 0, 0, 0]

    mppi_gym = base_mppi.BASE_MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, time_steps=TIMESTEPS,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    
    total_reward = base_mppi.run_mppi(mppi_gym, env, iter=500)

    env.reset()
    if downward_start:
        env.state = env.unwrapped.state = [0, 0, 0, 0]

    mppi_gym = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=10,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    

    total_reward = custom_mppi.run_mppi(mppi_gym, env, iter=500)
   


    env.close()