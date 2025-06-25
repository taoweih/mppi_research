import isaacsim
import isaaclab

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.mjcf import add_mjcf_asset

import numpy as np

world = World()
world.scene.add_default_ground_plane()

prim_path = "/World/go2"
mjcf_path = "mujoco_menagerie/unitree_go2/go2.xml"
add_mjcf_asset(prim_path,mjcf_path)

world.reset()
go2 = world.scene.get_object("go2")

for _ in range(10000):
    world.step(render=True)

simulation_app.close()


