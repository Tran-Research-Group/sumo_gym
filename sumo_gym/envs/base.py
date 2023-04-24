import os, sys
from abc import ABC, abstractmethod
from typing import Any, Final

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import traci.constants as tc

from sumo_gym.envs.types import InfoDict, ObsDict


class BaseSumoGymEnv(gym.Env, ABC):
    """
    The base Gymnasium environment for SUMO.
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
        ego_aware_dist: float = 100.0,
        ego_speed_mode: int = 32,
        vehicle_var_ids: list[int] = [tc.VAR_SPEED, tc.VAR_ANGLE, tc.VAR_POSITION],
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
        sumo_options: list[str]
            The list of SUMO options.
        ego_aware_dist: float
            The distance in meters that the ego vehicle is aware of.
        ego_speed_mode: int
            The speed mode of the ego vehicle.
        vehicle_var_ids: list[int]
            The list of vehicle variable IDs to obtain.
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

        self._num_actions: Final[int] = num_actions
        self._max_steps: Final[int] = max_steps
        self._config_path: Final[str] = config_path
        self._sumo_options: Final[list[str]] = sumo_options
        self._ego_aware_dist: Final[float] = ego_aware_dist
        self._ego_speed_mode: Final[int] = ego_speed_mode
        self._vehicle_var_ids: Final[list[int]] = vehicle_var_ids
        self._sumo_gui_binary: Final[str] = sumo_gui_binary
        self._sumo_binary: Final[str] = sumo_binary
        self._sumo_init_state_save_path: Final[str] = sumo_init_state_save_path
        self._is_gui_rendered: Final[bool] = is_gui_rendered
        self._reset_counter: int = 0

        self.action_space: gym.Space = spaces.Discrete(self._num_actions)
        self.observation_space: gym.Space = self._create_observation_space()

        self._sumo_cmd: list[str]
        if is_gui_rendered:
            self._sumo_cmd = (
                [sumo_gui_binary, "-c", self._config_path]
                + sumo_options
                + ["--start", "--quit-on-end"]
            )
        else:
            self._sumo_cmd = [
                sumo_binary,
                "-c",
                self._config_path,
                "--no-step-log",
                "true",
            ] + sumo_options

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

        if self._reset_counter > 0:
            self.close()
        else:
            pass

        self._step_count: int = 0
        self._reset_counter += 1

        traci.start(self._sumo_cmd)

        traci.vehicle.subscribeContext(
            "ego",
            tc.CMD_GET_VEHICLE_VARIABLE,
            self._ego_aware_dist,
            self._vehicle_var_ids,
        )
        traci.vehicle.setSpeedMode("ego", self._ego_speed_mode)

        traci.simulationStep()
        traci.simulation.saveState(self._sumo_init_state_save_path)
        self.state_dict: dict[
            str, dict[str, Any]
        ] = traci.vehicle.getContextSubscriptionResults("ego")
        self.vars: list[str] = list(self.state_dict["ego"].keys())
        self._ego_collided: bool = False
        self._ego_arrived_destination: bool = False

        observation: ObsDict = self._get_obs()
        self.observation: ObsDict = observation
        info: InfoDict = self._get_info()

        return observation, info

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
        self._act(action)
        self.state_dict = traci.vehicle.getContextSubscriptionResults("ego")

        self._step_count += 1

        terminated: bool = False
        truncated: bool = False
        if self._step_count >= self._max_steps:
            truncated = True
        else:
            pass

        observation: ObsDict = self._get_obs()
        self.observation = observation
        info: InfoDict = self._get_info()
        reward: float = self._reward()
        terminated: bool = self._terminate()

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _act(self, action: int) -> None:
        """
        Act the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        action: int
            The action to take.

        Returns
        --------
        None
        """
        ...

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
    def _terminate(self) -> bool:
        """
        Get whether the episode has terminated. This method needs be implemented in the child class.

        Parameters
        -----------
        None

        Returns
        --------
        terminated: bool
            Whether the episode has terminated.
        """
        ...

    def render(self, filename: str | None = None, mode: str = "human") -> None:
        """
        Render the environment. This method needs be implemented in the child class.

        Parameters
        -----------
        filename: str | None
            The filename to save the rendered environment.
        mode: str
            The mode to render the environment.

        Returns
        --------
        None
        """

        match filename:
            case None:
                pass
            case _:
                traci.gui.screenshot(traci.gui.DEFAULT_VIEW, filename)

    def close(self) -> None:
        traci.close()
