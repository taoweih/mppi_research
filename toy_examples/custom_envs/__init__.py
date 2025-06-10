from gymnasium.envs.registration import register
from .obstacle_avoidance import ObstacleAvoidanceEnv
from .cotinuous_acrobot import ContinuousAcrobotEnv

register(
    id='ObstacleAvoidance-v0',
    entry_point='custom_envs.obstacle_avoidance:ObstacleAvoidanceEnv',
)