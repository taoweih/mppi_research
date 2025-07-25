import argparse

import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.u_drone import UDrone


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":

    # Define the task (cost and dynamics)
    task = UDrone()

    # Set up the controller
    ctrl = MPPI(
        task,
        num_samples=512,
        noise_level=2.0,
        temperature=0.001,
        num_randomizations=1,
        plan_horizon=2.0,
        spline_type="zero",
        num_knots=4,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01

    mj_data = mujoco.MjData(mj_model)
    # mj_data.qpos[:] = mj_model.keyframe("hover").qpos
    # mj_data.ctrl[:] = mj_model.keyframe("hover").ctrl

    run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=25,
            show_traces=False,
            record_video=False,
        )
