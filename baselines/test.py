import airsimneurips as airsim 
import time
from baseline_racer import BaselineRacer

client = airsim.MultirotorClient()
client.confirmConnection()

client.simLoadLevel('Soccer_Field_Easy')
time.sleep(5.0)

baseline_racer_opp = BaselineRacer(drone_name='drone_2', viz_traj=True, viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0], viz_image_cv2=False)
baseline_racer_opp.initialize_drone()
baseline_racer_opp.start_race()
client.enableApiControl(vehicle_name='drone_1')
client.moveToPositionAsync(1, -1, 2, 1, vehicle_name='drone_1').join()


baseline_racer_opp.takeoff_with_moveOnSpline()
baseline_racer_opp.get_ground_truth_gate_poses()
baseline_racer_opp.fly_through_all_gates_at_once_with_moveOnSpline()

for gate_pose in baseline_racer_opp.gate_poses_ground_truth:
    print(gate_pose)
while True:
    #client.moveAndAvoid(vehicle_name='drone_1')
    client.flyAndAvoid([gate_pose.position for gate_pose in baseline_racer_opp.gate_poses_ground_truth], vel_max=30.0, acc_max=15.0, add_position_constraint=True, add_velocity_constraint=False, add_acceleration_constraint=False, replan_from_lookahead=False, vehicle_name='drone_1')