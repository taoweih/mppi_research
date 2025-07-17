import argparse

from evosax.algorithms.distribution_based.cma_es import CMA_ES

import mujoco

import numpy as np

from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an interactive simulation of the cube rotation task.

Double click on the floating target cube, then change the goal orientation with
[ctrl + left click].
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the cube rotation task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("mppi_staged_rollout", help="Model Predictive Path Integral Control with Staged Rollout")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.2,
        num_randomizations=32,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.4,
        temperature=0.001,
        num_randomizations=1,
        plan_horizon=3.00,
        spline_type="zero",
        num_knots=16,
    )
elif args.algorithm == "mppi_staged_rollout":
    print("Running MPPI with staged rollout")
    ctrl = MPPIStagedRollout(
        task,
        num_samples=128,
        noise_level=0.4,
        temperature=0.001,
        num_randomizations=8,
        plan_horizon=1.00,
        spline_type="zero",
        num_knots=16,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=5,
        sigma_start=0.5,
        sigma_min=0.5,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=128,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
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

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    fixed_camera_id=None,
    show_traces=False,
    max_traces=1,
    trace_color=[1.0, 1.0, 1.0, 1.0],
)
