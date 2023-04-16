import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci

from sumo_gym.envs.base import BaseSumoGymEnv
from sumo_gym.envs.types import InfoDict, ObsDict


class RoundaboutEnv(BaseSumoGymEnv):
    """
    The roundabout environment.
    """

    def __init__(
        self,
        num_veh: int,
        num_actions: int,
        max_steps: int,
        config_path: str,
        sumo_gui_binary: str = "/usr/bin/sumo-gui",
        sumo_binary: str = "/usr/bin/sumo",
        render: bool = False,
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
        super().__init__(
            num_veh,
            num_actions,
            max_steps,
            config_path,
            sumo_gui_binary,
            sumo_binary,
            render,
        )
