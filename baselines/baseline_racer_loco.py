import gtp
from baseline_racer import BaselineRacer
from utils import to_airsim_vector, to_airsim_vectors
from visualize import *

# Use non interactive matplotlib backend
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import airsimneurips as airsim
import time
import numpy as np

import argparse
import pdb
import threading
import os
import glob
import pdb

class BaselineRacerGTP(BaselineRacer):
    def __init__(self, traj_params, drone_names, drone_i, drone_params,
                 use_vel_constraints=False,
                viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], plot_gtp=False):
        super().__init__(drone_name=drone_names[drone_i], viz_traj=viz_traj, viz_traj_color_rgba=viz_traj_color_rgba, viz_image_cv2=False)
        self.drone_names = drone_names
        self.drone_i = drone_i
        self.drone_params = drone_params
        self.traj_params = traj_params

        self.use_vel_constraints = use_vel_constraints
        self.plot_gtp = plot_gtp

        self.controller = None
        self.replan_every_sec = 1.0
        self.check_log_file_every_sec = 0.1
        self.replan_callback_thread = threading.Thread(target=self.repeat_timer_replan, args=(self.replan_callback, self.replan_every_sec))
        self.log_monitor_thread = threading.Thread(target=self.repeat_timer_log_monitor, args=(self.log_monitor_callback, self.check_log_file_every_sec))
        self.timer_callback_thread = threading.Thread(target=self.repeat_timer, args=(self.timer_callback, 0.1))
        self.timer_callback_thread.daemon = True
        self.is_replan_thread_active = False
        self.is_log_monitor_thread_active = False

        self.airsim_client_replan_thread = airsim.MultirotorClient()
        self.airsim_client_replan_thread.confirmConnection()

        # For plotting: Just some fig, ax and line objects to keep track of
        if self.plot_gtp:
            self.fig, self.ax = plt.subplots()
            self.line_state = None
            self.lines = [None] * 2

        self.racing_log_path = None
        self.should_reset_race = False
        self.ignore_log_monitor_callback = False
        self.ignore_replan_callback = False
        self.raceEnded = False
        #self.ignore_threads = False
        self.ignore_main_thread = False

    def set_path_to_race_log_dir(self, path_to_race_log_dir):
        self.racing_log_path = path_to_race_log_dir

    def update_and_plan(self):
        print("REPLAN by Dr. Taubner")
        # Retrieve the current state from AirSim
        position_airsim = []
        for drone_name in self.drone_names:
            position_airsim.append(self.airsim_client_replan_thread.simGetObjectPose(drone_name).position)

        state = np.array([position.to_numpy_array() for position in position_airsim])

        if self.plot_gtp:
            print(state)
            # Plot or update the state
            if self.line_state is None:
                self.line_state, = plot_state(self.ax, state)
            else:
                replot_state(self.line_state, state)

        trajectory = self.controller.iterative_br(self.drone_i, state)

        # Now, let's issue the new trajectory to the trajectory planner
        # Fetch the current state first, to see, if our trajectory is still planned for ahead of us
        
        new_state_i = self.airsim_client_replan_thread.simGetObjectPose(self.drone_name).position.to_numpy_array()
        print(new_state_i)

        if self.plot_gtp:
            replot_state(self.line_state, state)

        # As we move while computing the trajectory,
        # make sure that we only issue the part of the trajectory, that is still ahead of us
        k_truncate, t = self.controller.truncate(new_state_i, trajectory[:, :])

        # print("k_truncate", k_truncate)

        # k_truncate == args.n means that the whole trajectory is behind us, and we only issue the last point
        if k_truncate == self.traj_params.n:
            k_truncate = self.traj_params.n - 1

        if self.plot_gtp:
            # For our 2D trajectory, let's plot or update
            if self.lines[self.drone_i] is None:
                self.lines[self.drone_i], = plot_trajectory_2d(self.ax, trajectory[k_truncate:, :])
            else:
                replot_trajectory_2d(self.lines[self.drone_i], trajectory[k_truncate:, :])

        k_truncate_size = 5

        if (trajectory.shape[0] - k_truncate_size) > 0:
            path = trajectory[k_truncate_size:, :]
        else:
            path = trajectory[-1,:]
        # k_truncate_idx = max((trajectory.shape(0)-k_truncate_size), 0)
        # Finally issue the command to AirSim.
        if not self.use_vel_constraints:
            # This returns a future, that we do not call .join() on, as we want to re-issue a new command
            # once we compute the next iteration of our high-level planner

            list_of_positions = to_airsim_vectors(path)

            print(list_of_positions)
            self.airsim_client_replan_thread.moveOnSplineAsync(to_airsim_vectors(path),
                                                 add_position_constraint=True,
                                                 add_velocity_constraint=False,
                                                 vel_max=self.drone_params[self.drone_i]["v_max"],
                                                 acc_max=self.drone_params[self.drone_i]["a_max"],
                                                 viz_traj=True, 
                                                 viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0] , replan_from_lookahead=True, replan_lookahead_sec=2.0, vehicle_name=self.drone_name)
        else:
            # Compute the velocity as the difference between waypoints
            vel_constraints = np.zeros_like(trajectory[k_truncate:, :])
            vel_constraints[1:, :] = trajectory[k_truncate + 1:, :] - trajectory[k_truncate:-1, :]
            # If we use the whole trajectory, the velocity constraint at the first point
            # is computed using the current position
            if k_truncate == 0:
                vel_constraints[0, :] = trajectory[k_truncate, :] - new_state_i
            else:
                vel_constraints[0, :] = trajectory[k_truncate, :] - trajectory[k_truncate - 1, :]

            vel_constraints /= self.traj_params.dt

            self.airsim_client_replan_thread.moveOnSplineVelConstraintsAsync(to_airsim_vectors(trajectory[k_truncate_size:, :]),
                                                               to_airsim_vectors(vel_constraints),
                                                               add_position_constraint=True,
                                                               add_velocity_constraint=True,
                                                               vel_max=self.drone_params[self.drone_i]["v_max"],
                                                               acc_max=self.drone_params[self.drone_i]["a_max"],
                                                               viz_traj=True, replan_from_lookahead=True, replan_lookahead_sec=1.0,  
                                                               vehicle_name=self.drone_name)

        print ("Dr. Taubner is done")
        if self.plot_gtp:
            # Refresh the updated plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def initialize_gtp_controller(self):
        self.get_ground_truth_gate_poses()

        # We pretend we have two different controllers for the drones,
        # so let's instantiate two
        self.controller = gtp.IBRController(self.traj_params, self.drone_params, self.gate_poses_ground_truth)

    def open_latest_log_file(self):
        all_log_files = glob.glob(os.path.join(self.racing_log_path, '*.log'))
        log_file_name = max(all_log_files, key=os.path.getctime) # get the latest log file
        print("Opened file: " + log_file_name)
        self.log_file = open(log_file_name, "r+")
        self.raceEnded = False

    def process(self, line):
        tokens = line.split()
        if len(tokens) != 3:
            print("ERROR Bad line: " + line)
            print("Tokens: " + str(tokens))
            return
        if not self.raceEnded:
            if tokens[1] == "disqualified" and tokens[2] == '1':
                print("disqualified")
                self.should_reset_race = True
                self.raceEnded = True
                return

            if tokens[1] == "finished" and tokens[2] == '1':
                print("finished")
                self.should_reset_race = True
                self.raceEnded = True
                return

    def follow(self, filename):
        filename.seek(0,2)
        while(True and self.is_log_monitor_thread_active):
            line = filename.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line  

    def log_monitor_callback(self):
        for line in self.follow(self.log_file):
            self.process(line)

    def repeat_timer_log_monitor(self, task, period):
        while self.is_log_monitor_thread_active:
            if not (self.ignore_replan_callback):
                task()
                time.sleep(period)

    def repeat_timer(self, task, period):
        while True: 
            task()
            time.sleep(period)

    def timer_callback(self):
        if(self.should_reset_race):
            #self.stop_log_monitor_callback_thread()
            #self.ignore_threads = True
            self.ignore_replan_callback = True
            self.airsim_client.clearTrajectory(vehicle_name=self.drone_names[0])
            self.airsim_client.clearTrajectory(vehicle_name=self.drone_names[1])
            # self.airsim_client.cancelLastTask(vehicle_name='drone_1')
            # self.airsim_client.cancelLastTask(vehicle_name='drone_2')
            #pdb.set_trace()
            #self.stop_replan_callback_thread()
            self.reset_race()
            time.sleep(2)
            self.airsim_client.enableApiControl()
            time.sleep(2)
            self.airsim_client.enableApiControl(self.drone_names[1])
            time.sleep(2)
            self.disarm_drone()
            time.sleep(2)
            self.airsim_client.disarm(self.drone_names[1])
            time.sleep(2)
            self.reset_race()
            self.should_reset_race = False
            #self.ignore_threads = False
            self.ignore_main_thread = False
            time.sleep(2)
            self.ignore_replan_callback = False

    def repeat_timer_replan(self, task, period):
        while self.is_replan_thread_active:
            if not (self.ignore_replan_callback):
                task()
                time.sleep(period)

    def start_replan_callback_thread(self):
        if not self.is_replan_thread_active:
            self.is_replan_thread_active = True
            self.replan_callback_thread.start()
            print("Started replanning thread")

    def stop_replan_callback_thread(self):
        if self.is_replan_thread_active:
            self.is_replan_thread_active = False
            self.replan_callback_thread.join()
            print("Stopped replanning thread.")

    def start_log_monitor_callback_thread(self):
        if not self.is_log_monitor_thread_active:
            self.is_log_monitor_thread_active = True
            self.log_monitor_thread.start()
            print("Started log monitor thread")

    def stop_log_monitor_callback_thread(self):
        if self.is_log_monitor_thread_active:
            self.is_log_monitor_thread_active = False
            self.log_monitor_thread.join()
            print("Stopped log monitor thread.")

    def replan_callback(self):
        if self.plot_gtp:
            # Let's plot the gates, and the fitted track.
            plot_gates_2d(self.ax, self.gate_poses_ground_truth)
            plot_track(self.ax, self.controller.track)
            plot_track_arrows(self.ax, self.controller.track)
            plt.show()

        if self.airsim_client_replan_thread.isApiControlEnabled(vehicle_name=self.drone_name):
            self.update_and_plan()

def main(args):
    drone_names = ["drone_1", "drone_2"]

    drone_params = [
        {"r_safe": 0.4,
         "r_coll": 0.3,
         "v_max": 15.0,
         "a_max": 10.0},
        {"r_safe": 0.4,
         "r_coll": 0.3,
         "v_max": 15.0,
         "a_max": 10.0}]

    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    gtp_racer = BaselineRacerGTP(
        traj_params=args,
        drone_names=drone_names,
        drone_i=0,
        drone_params=drone_params, 
        viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 1.0, 1.0],
        use_vel_constraints=args.vel_constraints, plot_gtp=args.plot_gtp)

    #baseline_racer_opp = BaselineRacer(drone_name=drone_names[1], viz_traj=True, viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0], viz_image_cv2=False)

    gtp_racer.set_path_to_race_log_dir("C:\\Users\\savempra\\Sim\\env\\NeuripsEnv\\Saved\\Logs\\RaceLogs")
    gtp_racer.load_level(args.level_name)
    gtp_racer.timer_callback_thread.start()

    gtp_racer.initialize_drone()
    #baseline_racer_opp.initialize_drone()

    gtp_racer.start_race(args.race_tier)
    gtp_racer.initialize_gtp_controller()
    #baseline_racer_opp.takeoff_with_moveOnSpline(0.3)
    #gtp_racer.takeoff_with_moveOnSpline(0.1)
    #baseline_racer_opp.get_ground_truth_gate_poses()
    #baseline_racer_opp.fly_through_all_gates_at_once_with_moveOnSpline()
    while True:
        gtp_racer.airsim_client.moveAndAvoid()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--vel_constraints', dest='vel_constraints', action='store_true', default=False)
    parser.add_argument('--plot_gtp', dest='plot_gtp', action='store_true', default=False)
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"], default="Soccer_Field_Easy")
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=3)
    main(parser.parse_args())
