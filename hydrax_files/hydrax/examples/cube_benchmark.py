import argparse

import mujoco
from tqdm import tqdm
import numpy as np

from hydrax.algs import  MPPI,  MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive, run_benchmark
from hydrax.tasks.cube import CubeRotation


if __name__ == "__main__":
    success = np.zeros(5)
    for h in tqdm(range(1, 5+1)):
        HORIZON = h*0.5

        # Define the task (cost and dynamics)
        task = CubeRotation()

        # Set up the controller
        ctrl = MPPI(
            task,
            num_samples=512,
            noise_level=0.4,
            temperature=0.001,
            num_randomizations=1,
            plan_horizon=HORIZON,
            spline_type="zero",
            num_knots=16,
        )

        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
        cube = mj_data.joint('cube_freejoint')
        uvw = np.random.rand(3)
        start_quat = np.array(
            [
                np.sqrt(1-uvw[0])*np.sin(2*np.pi*uvw[1]),
                np.sqrt(1-uvw[0])*np.cos(2*np.pi*uvw[1]),
                np.sqrt(uvw[0])*np.sin(2*np.pi*uvw[2]),
                np.sqrt(uvw[0])*np.cos(2*np.pi*uvw[2]),
            ]
        )
        cube.qpos[3:7] = start_quat


        num_success = run_benchmark(
            ctrl,
            mj_model,
            mj_data,
            frequency=25,
            GOAL_THRESHOLD=1,
        )
        print(num_success)
        success[h-1] = num_success
    print(success)

