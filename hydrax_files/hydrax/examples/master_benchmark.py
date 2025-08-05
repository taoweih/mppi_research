import mujoco

from hydrax.algs import MPPI, MPPIStagedRollout, PredictiveSampling
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


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":

    NUM_SAMPLES = 1024
    NOISE_LEVEL = 0.2
    TEMPERATURE = 0.001
    NUM_KNOTS = 16
    SPLINE_TYPE = "zero"

    NUM_KNOTS_PER_STAGE = 4

    success = np.zeros(10)
    for h in tqdm(range(1, 10+1)):
        HORIZON = h*0.2

        task_list = [UPointMass(), UDrone(), CubeRotation(), HumanoidStandup(), UR5e(), ArmReaching(), PushT(), Ant()]

        for i in range(len(task_list)):
            task = task_list[i]
            ctrl_list = [MPPI(task,num_samples=NUM_SAMPLES,noise_level=NOISE_LEVEL,temperature=TEMPERATURE,
                              num_randomizations=1,plan_horizon=HORIZON,spline_type=SPLINE_TYPE,num_knots=NUM_SAMPLES), 

                         MPPIStagedRollout(task, num_samples=NUM_SAMPLES, noise_level=NOISE_LEVEL, temperature=TEMPERATURE, 
                                           num_knots_per_stage=NUM_KNOTS_PER_STAGE, plan_horizon= HORIZON, spline_type=SPLINE_TYPE, num_knots=NUM_KNOTS),

                         PredictiveSampling()]
            for j in range(len(ctrl_list)):
                ctrl = ctrl_list[j]

                mj_model = task.mj_model
                mj_model.opt.timestep = 0.01

                mj_data = mujoco.MjData(mj_model)

                run_interactive(
                        ctrl,
                        mj_model,
                        mj_data,
                        frequency=50,
                        show_traces=False,
                        record_video=False,
                    )

    # # Set up the controller
    # ctrl = MPPI(
    #     task,
    #     num_samples=512,
    #     noise_level=0.2,
    #     temperature=0.001,
    #     num_randomizations=1,
    #     plan_horizon=2.0,
    #     spline_type="zero",
    #     num_knots=16,
    # )
