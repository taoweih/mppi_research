import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, PredictiveSampling, DIAL, CEM
from hydrax.simulation.deterministic import run_interactive, run_benchmark

from hydrax.tasks.u_point_mass import UPointMass
from hydrax.tasks.u_drone import UDrone
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.humanoid_standup import HumanoidStandup
from hydrax.tasks.ur5e import UR5e
from hydrax.tasks.arm_reaching import ArmReaching
from hydrax.tasks.pusht import PushT
from hydrax.tasks.ant import Ant

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from hydrax import ROOT
from datetime import datetime
from pathlib import Path
import os


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":

    NUM_SAMPLES = 1024
    NOISE_LEVEL = 0.2
    TEMPERATURE = 0.001
    NUM_KNOTS = 16
    SPLINE_TYPE = "zero"

    NUM_KNOTS_PER_STAGE = 4

    success = np.zeros((8, 3, 10))



    task_list = [UPointMass(), UDrone(), CubeRotation(), HumanoidStandup(), UR5e(), ArmReaching(), PushT(), Ant()]

    for i in range(len(task_list)):
        task = task_list[i]

        for h in range(10):
            HORIZON = (h+1)*0.2

            ctrl_list = [MPPI(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL,temperature=TEMPERATURE
                              ,plan_horizon=HORIZON,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS), 

                        MPPIStagedRollout(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE, 
                                        num_knots_per_stage=NUM_KNOTS_PER_STAGE, plan_horizon= HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),

                        PredictiveSampling(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, plan_horizon=HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS)]
            
            for j in range(len(ctrl_list)):
                ctrl = ctrl_list[j]

                mj_model = task.mj_model
                mj_model.opt.timestep = 0.01

                mj_data = mujoco.MjData(mj_model)

                # num_success = run_benchmark(
                #     ctrl,
                #     mj_model,
                #     mj_data,
                #     frequency=25,
                #     GOAL_THRESHOLD=1,
                # )
                num_success = h+j

                success[i, j, h] = num_success

    # Plotting
    curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = Path(ROOT)/"benchmark"/f"run_{curr_time}"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(save_dir, "params.txt")

    params = f'''
        MPPI:
            Number of samples: {NUM_SAMPLES}
            Noise level: {NOISE_LEVEL}
            Temperature: {TEMPERATURE}
            Horizon: {HORIZON}s
            Spline type: {SPLINE_TYPE}
            Number of knots: {NUM_KNOTS}
        MPPI staged rollout:
            Number of samples: {NUM_SAMPLES}
            Noise level: {NOISE_LEVEL}
            Temperature: {TEMPERATURE}
            Horizon: {HORIZON}s
            Spline type: {SPLINE_TYPE}
            Number of knots: {NUM_KNOTS}
            Number of knots per stage: {NUM_KNOTS_PER_STAGE}
        PS:
            Number of samples: {NUM_SAMPLES}
            Noise level: {NOISE_LEVEL}
            Horizon: {HORIZON}s
            Spline type: {SPLINE_TYPE}
            Number of knots: {NUM_KNOTS}
        '''

    with open(file_path,'w') as f:
        f.write(params)
    f.close()

    for i in range(success.shape[0]):
        plt.figure()

        for j in range(success.shape[1]):
            plt.plot(np.linspace(0.2, 2.0, 10), success[i, j], label=type(ctrl_list[j]).__name__)

        plt.title(f'Task {type(task_list[i]).__name__}')
        plt.xlabel("Horizon (seconds)")
        plt.ylabel("Sucess Rate (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"task_{type(task_list[i]).__name__}.png", dpi=300)
        plt.close()
        

