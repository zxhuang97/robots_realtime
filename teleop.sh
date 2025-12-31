python dependencies/i2rt/examples/teleop_record_replay.py --mode dual_follower --gripper crank_4310 --left-follower-port 11333 --right-follower-port 11334 --output ./teleop_trajectories

python dependencies/i2rt/examples/teleop_record_replay.py --mode dual_leader --gripper yam_teaching_handle --bilateral-kp 0.01 --left-follower-port 11333 --right-follower-port 11334 --output ./teleop_trajectories

