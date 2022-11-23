from tensorforce.environments import Environment

environment = Environment.create(
    environment='terrain', level='CartPole', max_episode_timesteps=500
)
