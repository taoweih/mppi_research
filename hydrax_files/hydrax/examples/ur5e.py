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
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        num_randomizations=1,
        plan_horizon=0.2,
        spline_type="zero",
        num_knots=16,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.005

    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("home").qpos

    run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
            record_video=False,
        )
