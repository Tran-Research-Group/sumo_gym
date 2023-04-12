from sumoEnv import sumoEnv
import traci
import traci.constants as tc

n_veh=5
num_actions=11                                  #0 m/s, 1 m/s,..., 10 m/s
num_s=24   
env=sumoEnv(n_veh,num_actions,num_s,render=True)                                    #(speed,heading,2dPos) for 6 veh total

# traci.vehicle.setSpeedMode("ego",0)
traci.vehicle.subscribeContext("ego", tc.CMD_GET_VEHICLE_VARIABLE, 200, [tc.VAR_SPEED, tc.VAR_ANGLE,tc.VAR_POSITION])
traci.simulationStep()
sub=traci.vehicle.getContextSubscriptionResults("ego")
vars=list(sub["ego"].keys())

env.addContextVars(vars)
step = 1
while step < 1000:
   s,r,done=env.step()
   print(r)
#    traci.simulationStep()
#    sub=traci.vehicle.getContextSubscriptionResults("ego")
#    if len(sub)>0:
#       s=env.construct_state(sub)
#    traci.vehicle.setSpeed("ego",8.5)
   step += 1

traci.close()