import argparse

import mujoco
import numpy as np
from tqdm import tqdm

from hydrax.algs import MPPI, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive, run_benchmark
from hydrax.tasks.humanoid_standup import HumanoidStandup



if __name__ == "__main__":
    success = np.zeros(10)
    for h in tqdm(range(1, 10+1)):
        HORIZON = h*0.1

        # Define the task (cost and dynamics)
        task = HumanoidStandup()

        # Set up the controller
        ctrl = MPPI(
            task,
            num_samples=1024,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=1,
            plan_horizon=HORIZON,
            spline_type="zero",
            num_knots=16,
        )

        # Define the model used for simulation (stiffer contact parameters)
        mj_model = task.mj_model
        mj_model.opt.timestep = 0.005
        mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
        mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

        # Set the initial state so the robot falls and needs to stand back up
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = mj_model.keyframe("stand").qpos
        mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]
        mj_data.qpos[2] = 0.1

        num_success = run_benchmark(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            GOAL_THRESHOLD=11,
        )
        print(num_success)
        success[h-1] = num_success
    print(success)
