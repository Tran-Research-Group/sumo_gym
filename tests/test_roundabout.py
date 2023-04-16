import unittest

from sumo_gym.envs.roundabout import RoundaboutEnv


class TestRoundaboutEnv(unittest.TestCase):
    def test_roundabout_env(self):
        env = RoundaboutEnv(5, 11, 1000, "configs/roundabout/roundabout.sumocfg")
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
        env.close()
