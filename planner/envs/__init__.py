from gym.envs.registration import register
# add all custom envs here

register(
    id='PlannerEnv-v0',
    entry_point='planner.envs.plannerEnvs_dir:PlannerEnv'
)
