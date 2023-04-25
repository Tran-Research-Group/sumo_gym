import unittest

from sumo_gym.envs.roundabout import RoundaboutEnv


class TestRoundaboutEnv(unittest.TestCase):
    def test_roundabout_env(self):
        env = RoundaboutEnv(
            20,
            300,
            "sumo_gym/config/roundabout/roundabout.sumocfg",
            sumo_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            sumo_gui_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            is_gui_rendered=True,
        )
        env.reset()
        for i in range(100):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            env.render(f"out/roundabout/test_roundabout_env_{i}.png")
            if terminated or truncated:
                break
