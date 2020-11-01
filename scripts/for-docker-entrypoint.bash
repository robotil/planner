. /opt/ros/eloquent/setup.bash
cd /home/iaiai/git/ros2ws/
colcon build
echo install/setup.bash >> ~/.bashrc
cd /home/iaiai/git/ros2ws/src/planner
cp -r logic_simulator  /home/iaiai/git/ros2ws/install/planner/lib/python3.6/site-packages/
cp -r planner/envs  /home/iaiai/git/ros2ws/install/planner/lib/python3.6/site-packages/planner