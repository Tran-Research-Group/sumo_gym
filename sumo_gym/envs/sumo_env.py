import traci
import traci.constants as tc
import os, sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from numpy import linalg as la
import numpy as np


class SumoEnv:
    def __init__(self, n_veh, num_actions, num_s, render=False):
        """
        self.n_veh: number of surrounding vehicles in state vector
        """
        self.n_veh = n_veh
        self.num_actions = num_actions
        self.s = []
        self.num_s = num_s
        sumoBinary = "/home/niket/sumo/bin/sumo-gui"
        if render:
            self.sumoCmd = [sumoBinary, "-c", "tjunction.sumocfg"]
        else:
            self.sumoCmd = [
                "/home/niket/sumo/bin/sumo",
                "-c",
                "tjunction.sumocfg",
                "--no-step-log",
                "true",
            ]
        traci.start(self.sumoCmd)

    def construct_state(self, statedict):
        """
        Caution: else part is hardcoded (in terms of elements of state and their order) when adding dummy values
        """
        self.statedict = statedict
        self.context_len = len(self.statedict) - 1
        if self.context_len > self.n_veh:
            distances = {}
            ego_pos = np.array(self.statedict["ego"][self.vars[0]])
            for vehicle in self.statedict:
                veh_pos = np.array(self.statedict[vehicle][self.vars[0]])
                distances[vehicle] = la.norm(ego_pos - veh_pos)
            sorted_distances = {
                k: v for k, v in sorted(distances.items(), key=lambda item: item[1])
            }
            s = []
            for vehicle in list(sorted_distances.keys())[: self.n_veh + 1]:
                v_s_dict = self.statedict[vehicle]
                v_s_list = [v_s_dict[key] for key in v_s_dict.keys()]
                v_s_list = np.concatenate(v_s_list, axis=None)
                s.append(v_s_list)
            self.s = np.concatenate(s, axis=None)
        else:
            s = []
            for vehicle in self.statedict:
                v_s_dict = self.statedict[vehicle]
                v_s_list = [v_s_dict[key] for key in v_s_dict.keys()]
                v_s_list = np.concatenate(v_s_list, axis=None)
                s.append(v_s_list)
            s = np.concatenate(s, axis=None)

            if self.context_len < self.n_veh:
                n_dummy = self.n_veh - self.context_len
                dummy_state = [0, 0, 10000, 10000]
                dummy_state_cat = dummy_state
                for _ in range(n_dummy - 1):
                    dummy_state_cat = [*dummy_state_cat, *dummy_state]
                s = [*s, *dummy_state_cat]
            self.s = np.array(s)

        # return self.s

    # def startSim(self):
    #      sumoBinary = "/home/niket/sumo/bin/sumo-gui"
    #      sumoCmd = [sumoBinary, "-c", "tjunction.sumocfg"]
    #      self.traci.start(sumoCmd)

    # def closeSim(self):
    #      self.traci.close()

    # def subscribeContext(self):
    #      self.traci.vehicle.subscribeContext("ego", tc.CMD_GET_VEHICLE_VARIABLE, 200, [tc.VAR_SPEED, tc.VAR_ANGLE,tc.VAR_POSITION])

    # def getContextVars(self): # this part gets keys for state dict. requires ego starting at time 0.
    #      self.traci.simulationStep()
    #      sub=self.traci.vehicle.getContextSubscriptionResults("ego")
    #      print(sub)
    #      return list(sub["ego"].keys())

    def addContextVars(self, var_keys):
        self.vars = var_keys

    def step(self):
        traci.simulationStep()
        sub = traci.vehicle.getContextSubscriptionResults("ego")
        done = False
        print(sub)
        if len(sub) > 0:
            self.construct_state(sub)
            r = self.compute_reward(self.s)
            if self.s[2] < 290:
                done = True
            return self.s, r, done
        else:
            done = True
            r = 0
            return self.s, r, done

    def compute_reward(self, s):
        x_ego = s[2]
        x_goal = 292  # hardcoded right now
        if x_ego < x_goal:
            return 10
        else:
            return -0.1

    def reset(self):
        traci.close(False)
        traci.start(self.sumoCmd)
        self.s = []
        # traci.load(["-c", "tjunction.sumocfg","--no-step-log", "true"])
        # return self.get_observation()
