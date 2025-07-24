import argparse

import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.ur5e import UR5e


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":

    # Define the task (cost and dynamics)
    task = UR5e()

    # Set up the controller
    ctrl = MPPIStagedRollout(
        task,
        num_samples=1024,
        noise_level=0.4,
        temperature=0.01,
        num_randomizations=1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=16,
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
            frequency=25,
            show_traces=False,
            record_video=False,
        )
