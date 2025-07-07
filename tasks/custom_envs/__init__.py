from gymnasium.envs.registration import register
from .obstacle_avoidance import ObstacleAvoidanceEnv
from .continuous_acrobot import ContinuousAcrobotEnv

register(
    id='ObstacleAvoidance-v0',
    entry_point='custom_envs.obstacle_avoidance:ObstacleAvoidanceEnv',
)
register(
    id='ContinuousAcrobot-v1',
    entry_point='custom_envs.continuous_acrobot:ContinuousAcrobotEnv',
)