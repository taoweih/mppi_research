# import torch
# import isaacsim
# from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False})
# from omni.isaac.lab.sim import SimulationContext
# from omni.isaac.lab.envs import make

# import time

# env = make("Isaac-Cartpole-v0")

# env.reset()

# nu = env.action_shape[0]
# action = torch.zeros((1,nu), device=env.device)

# for _ in range(200000):
#     env.step(action)

#     env.render()
#     time.sleep(env.sim.dt)

# env.close()
# simulation_app.close()


from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    from omni.isaac.lab.sim import SimulationContext
    print("✅ omni.isaac.lab is available!")
except Exception as e:
    print("❌ Failed to import omni.isaac.lab:", e)
app.close()


