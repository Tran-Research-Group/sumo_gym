from typing import Any, Final, TypedDict

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import traci.constants as tc

from sumo_gym.envs.base import BaseSumoGymEnv

from sumo_gym.envs.types import InfoDict, ObsDict


class RoundaboutEnv(BaseSumoGymEnv):
    """
    The roundabout environment.
    """

    def __init__(
        self,
        num_actions: int,
        max_steps: int,
        config_path: str,
        sumo_options: list[str] = [
            "--step-length",
            "0.1",
            "--collision.check-junctions",
            "true",
            "--collision.action",
            "warn",
            "--emergencydecel.warning-threshold",
            "1000.1",
        ],
        max_ego_speed: float = 10.0,  # m/s
        ego_aware_dist: float = 100.0,
        ego_speed_mode: int = 32,
        sumo_gui_binary: str = "/usr/bin/sumo-gui",
        sumo_binary: str = "/usr/bin/sumo",
        sumo_init_state_save_path: str = "out/sumoInitState.xml",
        is_gui_rendered: bool = False,
    ) -> None:
        """
        Initialize the environment.

        Parameters
        -----------
        num_veh: int
            The number of surrounding vehicles in the state vector.
        num_actions: int
            The number of actions available to the agent.
        max_steps: int
            The maximum number of steps the agent can take in the environment.
        config_path: str
            The path to the SUMO configuration file.
        sumo_gui_binary: str
            The path to the SUMO GUI binary.
        sumo_binary: str
            The path to the SUMO binary.
        render: bool
            Whether to render the environment.

        Returns
        --------
        None

        """
        vehicle_var_ids: list[int] = [tc.VAR_SPEED, tc.VAR_POSITION]
        self._max_ego_speed: Final[float] = max_ego_speed
        super().__init__(
            num_actions,
            max_steps,
            config_path,
            sumo_options,
            ego_aware_dist,
            ego_speed_mode,
            vehicle_var_ids,
            sumo_gui_binary,
            sumo_binary,
            sumo_init_state_save_path,
            is_gui_rendered,
        )

    def _create_observation_space(self) -> gym.Space:
        observation_space = spaces.Dict(
            {
                "ego_speed": spaces.Box(
                    low=0.0, high=30.0, shape=(1,), dtype=np.float32
                ),
                "ego_pos": spaces.Box(
                    low=0.0, high=100.0, shape=(2,), dtype=np.float32
                ),
                "t_0_speed": spaces.Box(
                    low=0.0, high=30.0, shape=(1,), dtype=np.float32
                ),
                "t_0_pos": spaces.Box(
                    low=0.0, high=100.0, shape=(2,), dtype=np.float32
                ),
                "t_1_speed": spaces.Box(
                    low=0.0, high=30.0, shape=(1,), dtype=np.float32
                ),
                "t_1_pos": spaces.Box(
                    low=0.0, high=100.0, shape=(2,), dtype=np.float32
                ),
            }
        )

        return observation_space

    def _get_obs(self) -> ObsDict:
        state_dict: dict[str, dict[str, Any]] = self.state_dict

        observation = {}

        for vehicle in self.state_dict:
            pos = np.array(state_dict[vehicle][self.vars[0]])
            vel = np.array(state_dict[vehicle][self.vars[1]])
            pos_key: str = vehicle + "_pos"
            vel_key: str = vehicle + "_speed"

            observation[pos_key] = pos
            observation[vel_key] = vel

        return observation

    def _get_info(self) -> InfoDict:
        pos_key: str = self.vars[0]
        ego_t0_dist = np.linalg.norm(
            self.state_dict["ego"][pos_key] - self.state_dict["t_0"][pos_key]
        )
        ego_t1_dist = np.linalg.norm(
            self.state_dict["ego"][pos_key] - self.state_dict["t_1"][pos_key]
        )

        info: InfoDict = {
            "ego_t0_dist": ego_t0_dist,
            "ego_t1_dist": ego_t1_dist,
            "ego_collided": self._ego_collided,
        }

        return info

    def _act(self, action: int) -> None:
        traci.vehicle.setSpeed(
            "ego", np.linspace(0.0, self._max_ego_speed, self._num_actions)[action]
        )

    def _terminate(self) -> bool:
        terminated: bool = self._ego_collided or self._ego_arrived_destination

        return terminated

    def _reward(self) -> float:
        reward: float = 0.0

        destination_x: float = -52.0

        self._ego_collided = "ego" in traci.simulation.getCollidingVehiclesIDList()

        if self._ego_collided:
            reward = -1.0
        else:
            pass

        if self.observation["ego_pos"][0] < destination_x:
            reward = 1.0
            self._ego_arrived_destination = True
        else:
            pass

        return reward
