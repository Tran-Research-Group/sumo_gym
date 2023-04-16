import os, sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci

from sumo_gym.envs.types import InfoDict, ObsDict


class BaseSumoGymEnv(gym.Env, ABC):
    """
    The base Gymnasium environment for SUMO.
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
        """
        super().__init__()
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            raise NameError(
                "[sumo_gym] Please declare environment variable 'SUMO_HOME'"
            )

        self._num_veh: int = num_veh
        self._num_actions: int = num_actions
        self._max_steps: int = max_steps
        self._config_path: str = config_path
        self._sumo_binary: str = sumo_binary
        self._render: bool = render

        self.action_space: gym.Space = spaces.Discrete(self._num_actions)
        self.observation_space: gym.Space = self._create_observation_space()

        self._sumo_cmd: list[str]
        if render:
            self._sumo_cmd = [sumo_gui_binary, "-c", self._config_path]
        else:
            self._sumo_cmd = [
                sumo_binary,
                "-c",
                self._config_path,
                "--no-step-log",
                "true",
            ]

    @abstractmethod
    def _create_observation_space(self) -> gym.Space:
        """
        Create the observation space. This method needs be implemented in the child class.

        Parameters
        -----------
        None

        Returns
        --------
        observation_space: gym.Space
            The observation space.
        """
        ...

    @abstractmethod
    def _get_obs(self) -> ObsDict:
        """
        Get the observation. This method needs be implemented in the child class.

        Parameters
        -----------
        None

        Returns
        --------
        observation: ObsDict
            The observation of the environment.
        """
        ...

    @abstractmethod
    def _get_info(self) -> InfoDict:
        """
        Get the info to diagnose the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        None

        Returns
        --------
        info: InfoDict
            The info to diagnose the environment.
        """
        ...

    @abstractmethod
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsDict, InfoDict]:
        """
        Reset the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        seed: int | None
            The seed to use for the environment.
        options: dict[str, Any] | None
            The options to use for the environment.

        Returns
        --------
        observation: ObsDict
            The observation of the environment.
        info: InfoDict
            The info of the environment.
        """
        super().reset(seed=seed, options=options)

        self._step_count: int = 0

        traci.start(self._sumo_cmd)

        observation: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        return observation, info

    @abstractmethod
    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]:
        """
        Step the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        action: int
            The action to take.

        Returns
        --------
        observation: ObsDict
            The observation of the environment.
        reward: float
            The reward from the environment.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated.
        info: InfoDict
            The info of the environment.
        """
        traci.simulationStep()
        self._step_count += 1

        terminated: bool = False
        truncated: bool = False
        if self._step_count >= self._max_steps:
            terminated = True
        else:
            pass

        reward: float = self._reward()
        observation: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _reward(self) -> float:
        """
        Get the reward. This method needs be implemented in the child class.

        Parameters
        -----------
        None

        Returns
        --------
        reward: float
            The reward from the environment.
        """
        ...

    @abstractmethod
    def render(self, mode: str = "human") -> None:
        """
        Render the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        mode: str
            The mode to render the environment.

        Returns
        --------
        None
        """
        ...
