import argparse

import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.drone import Drone


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":

    # Define the task (cost and dynamics)
    task = Drone()

    # Set up the controller
    ctrl = MPPI(
        task,
        num_samples=64,
        noise_level=0.2,
        temperature=0.001,
        num_randomizations=1,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=4,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01

    mj_data = mujoco.MjData(mj_model)
    # mj_data.qpos[:] = mj_model.keyframe("home").qpos

    run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
            record_video=False,
        )
