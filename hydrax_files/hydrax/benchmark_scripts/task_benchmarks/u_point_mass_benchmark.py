import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, PredictiveSampling, DIAL, CEM
from hydrax.simulation.deterministic import run_interactive, run_benchmark

from hydrax.tasks.u_point_mass import UPointMass

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from hydrax import ROOT
from datetime import datetime
from pathlib import Path
import os
import json


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # common parameters
    NUM_SAMPLES = 512
    NOISE_LEVEL = 2.0
    TEMPERATURE = 0.01
    NUM_KNOTS = 16
    SPLINE_TYPE = "zero"

    # MPPI staged rollout specific
    NUM_KNOTS_PER_STAGE = 4

    # DIAL specific
    BETA_OPT_ITER = 1
    BETA_HORIZON = 0.7

    # CEM specific
    NUM_ELITES = int(NUM_SAMPLES/4)
    SIGMA_START = NOISE_LEVEL/2
    SIGMA_MIN = NOISE_LEVEL/8
    EXPLORE_FRACTION = 0.5

    Horizon_steps = 13
    Horizon_start = 0.2
    Horizon_end = 2.0
    

    success = np.zeros((5, Horizon_steps)) # number of controlers by horizon
    all_frequency = np.zeros((5, Horizon_steps))
    all_state_trajectory = [[],[],[],[],[]] # number of controllers by horizon by number of iteration by number of trials by qpos shape
    all_control_trajectory = [[],[],[],[],[]]

    task = UPointMass()

    for h in tqdm(range(Horizon_steps)):
        HORIZON = (h+1)*0.1 + 0.7

        ctrl_list = [PredictiveSampling(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, plan_horizon=HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),
                     
                    MPPI(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL,temperature=TEMPERATURE
                            ,plan_horizon=HORIZON,spline_type=SPLINE_TYPE,num_knots=NUM_KNOTS), 

                    MPPIStagedRollout(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE, 
                                    num_knots_per_stage=NUM_KNOTS_PER_STAGE, plan_horizon= HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),

                    DIAL(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, beta_opt_iter=BETA_OPT_ITER, beta_horizon=BETA_HORIZON, temperature=TEMPERATURE, plan_horizon=HORIZON,
                         spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),

                    CEM(task,num_samples=NUM_SAMPLES, num_elites=NUM_ELITES, sigma_start=SIGMA_START, sigma_min=SIGMA_MIN, explore_fraction=EXPLORE_FRACTION, plan_horizon=HORIZON, 
                        spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS)]
        
        for j in range(len(ctrl_list)):
            ctrl = ctrl_list[j]

            mj_model = task.mj_model
            mj_model.opt.timestep = 0.01

            mj_data = mujoco.MjData(mj_model)

            num_success, control_freq, state_trajectory, control_trajectory = run_benchmark(
                ctrl,
                mj_model,
                mj_data,
                frequency=25,
                GOAL_THRESHOLD=0.05,
            )
            # num_success = h+j
            # control_freq = 0
            # state_trajectory = 0
            # control_trajectory = 0

            success[j, h] = num_success
            all_frequency[j, h] = control_freq
            all_state_trajectory[j].append(state_trajectory)
            all_control_trajectory[j].append(control_trajectory)

    #params
    curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = Path(ROOT)/"benchmark"/f"u_point_mass_benchmark_{curr_time}"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(save_dir, "params.json")

    params = {
        "PS": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS
        },
        "MPPI": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS
        },
        "MPPI staged rollout": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS,
            "Number of knots per stage": NUM_KNOTS_PER_STAGE
        },
        "DIAL": {
            "Number of samples": NUM_SAMPLES,
            "Noise level": NOISE_LEVEL,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS,
            "Beta opt iter": BETA_OPT_ITER,
            "Beta horizon": BETA_HORIZON
        },
        "CEM": {
            "Number of samples": NUM_SAMPLES,
            "Temperature": TEMPERATURE,
            "Horizon (s)": HORIZON,
            "Spline type": SPLINE_TYPE,
            "Number of knots": NUM_KNOTS,
            "Number of elites": NUM_ELITES,
            "Sigma start": SIGMA_START,
            "Sigma min": SIGMA_MIN,
            "Explore fraction": EXPLORE_FRACTION
        }
    }

    with open(file_path, "w") as f:
        json.dump(params, f, indent=4)

    # state and control trajectories
    file_path = os.path.join(save_dir, "trajectory.npz")
    all_state_trajectory = np.array(all_state_trajectory)
    all_control_trajectory = np.array(all_control_trajectory)
    np.savez(file_path, state_trajectory=all_state_trajectory, control_trajectory=all_control_trajectory)
    
    # control frequency
    file_path = os.path.join(save_dir, "frequency.csv")
    np.savetxt(file_path, all_frequency, delimiter=",",fmt="%.2e")

    plt.figure()
    for j in range(all_frequency.shape[0]):
        plt.plot(np.linspace(Horizon_start, Horizon_end, Horizon_steps), all_frequency[j], label=type(ctrl_list[j]).__name__)
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Control Frequency (HZ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"frequency.png", dpi=300)
    plt.close()


    # sucess rate
    file_path = os.path.join(save_dir, "sucess_count.csv")
    np.savetxt(file_path, (success).astype(int), delimiter=",",fmt="%d")

    plt.figure()
    for j in range(success.shape[0]):
        plt.plot(np.linspace(Horizon_start, Horizon_end, Horizon_steps), success[j], label=type(ctrl_list[j]).__name__)
    plt.title(f'Task {type(task).__name__}')
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Sucess Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"task_{type(task).__name__}.png", dpi=300)
    plt.close()