from gymnasium.envs.registration import register
from .obstacle_avoidance import ObstacleAvoidanceEnv

register(
    id='ObstacleAvoidance-v0',
    entry_point='custom_envs.obstacle_avoidance:ObstacleAvoidanceEnv',
)