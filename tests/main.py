from sumo_gym.envs.roundabout import RoundaboutEnv

env = RoundaboutEnv(20, 300, "sumo_gym/config/roundabout/roundabout.sumocfg")
env.reset()
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    env.render("out/roundabout/test_roundabout_env")
    if terminated or truncated:
        break
