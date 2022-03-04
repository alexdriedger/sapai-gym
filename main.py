from SuperAutoPetsEnv import SuperAutoPetsEnv


env = SuperAutoPetsEnv(print)
obs = env.reset()
n_steps = 1000
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
