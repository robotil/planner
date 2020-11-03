#Download base image ubuntu 18.04
FROM osrf/ros:eloquent-desktop
# LABEL about the custom image
LABEL maintainer="michele@robotics-iai.com"
LABEL version="0.1"
LABEL description="This is the Docker Image for \
the training of the planner's brain."

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# Complete installation from ubuntu repository
RUN apt install -y ros-eloquent-ros-base ros-eloquent-rmw-opensplice-cpp && \
    apt install -y vim python3-pip openssh-server python3-tk && \
    RTI_NC_LICENSE_ACCEPTED=yes apt-get install rti-connext-dds-5.3.1 && \
    apt install -y ros-eloquent-ament-cmake && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean

# Complete installation from python repository
RUN pip3 install --upgrade pip && \
    pip3 install stable-baselines[mpi] && \
    pip3 install stable-baselines[mpi] --upgrade && \
    pip3 install tensorflow==1.14  && \
    pip3 install keras==2.3.1 && \
    pip3 install pyproj

# Complete installation from GIT repository
RUN mkdir -p /home/iaiai/git/ros2ws/src && \
    cd /home/iaiai/git && \
    git clone https://github.com/mhallak/scripts.git && \
    cd /home/iaiai/git/ros2ws/src && \
    git clone https://github.com/robotil/planner_msgs.git && \
    git clone https://github.com/robotil/planner.git && \
    cp /home/iaiai/git/ros2ws/src/planner/scripts/for-docker-entrypoint.bash /for-docker-entrypoint.sh && \
    pip3 install pyproj

# Build project
ENV ROS_PYTHON_VERSION=3
ENV ROS_DISTRO=eloquent
ENV ROS_VERSION=2
ENV ROS_LOCALHOST_ONLY=0
#ENV ROS_DOMAIN_ID=2
  

CMD ["/bin/bash"]
ENTRYPOINT ["/bin/bash", "-c", "/for-docker-entrypoint.sh"]

