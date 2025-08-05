import mujoco
import argparse

from hydrax.algs import PredictiveSampling, MPPI, MPPIStagedRollout
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
task = PushT()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the pusht task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("mppi_staged_rollout", help="Model Predictive Path Integral Control with Staged Rollout")
args = parser.parse_args()

# Set up the controller
ctrl = PredictiveSampling(
    task,
    num_samples=128,
    noise_level=0.4,
    num_randomizations=4,
    plan_horizon=0.5,
    spline_type="zero",
    num_knots=6,
)

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.4,
        num_randomizations=4,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=10.0,
        num_randomizations=1,
        temperature=0.0001,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=16,
    )
elif args.algorithm == "mppi_staged_rollout":
    print("Running MPPI with staged rollout")
    ctrl = MPPIStagedRollout(
        task,
        num_samples=128,
        noise_level=10.0,
        num_randomizations=1,
        temperature=0.0001,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=16,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    max_traces=5,
)
