import unittest

from sumo_gym.envs.roundabout import RoundaboutEnv


class TestRoundaboutEnv(unittest.TestCase):
    def test_roundabout_env(self):
        env = RoundaboutEnv(
            20,
            300,
            r"C:\Users\mik09\Development\git\sumo_gym\sumo_gym\config\roundabout\roundabout.sumocfg",  # "./sumo_gym/config/roundabout/roundabout.sumocfg",
            sumo_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            sumo_gui_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            is_gui_rendered=True,
        )
        env.reset()
        for _ in range(1000):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            env.render("out/roundabout/test_roundabout_env")
            if terminated or truncated:
                break
