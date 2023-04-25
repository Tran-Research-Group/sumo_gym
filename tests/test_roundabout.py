import unittest

from sumo_gym.envs.roundabout import RoundaboutEnv


class TestRoundaboutEnv(unittest.TestCase):
    def test_roundabout_env(self):
        env = RoundaboutEnv(
            20,
            300,
            "sumo_gym/config/roundabout/roundabout.sumocfg",
            # sumo_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            # sumo_gui_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            is_gui_rendered=True,
            step_length=1,
            ego_aware_dist=200,
        )
        env.reset()
        for i in range(25):
            if i == 16:
                pass
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            env.render(f"out/roundabout/test_roundabout_env_{i}.png")
            if terminated or truncated:
                break

    def test_roundabout_env_no_gui(self):
        env = RoundaboutEnv(
            20,
            300,
            "sumo_gym/config/roundabout/roundabout.sumocfg",
            # sumo_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            # sumo_gui_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            step_length=1,
            ego_aware_dist=200,
        )
        env.reset()
        for i in range(25):
            if i == 16:
                pass
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break
