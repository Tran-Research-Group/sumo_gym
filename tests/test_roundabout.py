import unittest

from sumo_gym.envs.roundabout import RoundaboutEnv


class TestRoundaboutEnv(unittest.TestCase):
    def test_roundabout_env(self):
        env = RoundaboutEnv(
            20,
            300,
            "configs/roundabout/roundabout.sumocfg",
        )
        env.reset()
        for _ in range(1000):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            env.render("out/roundabout/test_roundabout_env")
            if terminated or truncated:
                break
