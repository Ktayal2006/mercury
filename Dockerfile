FROM osrf/ros:jazzy-desktop
SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    sudo \
    git \
    vim \
    neovim \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi && rosdep update

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# Face recognition dependencies
RUN pip3 install insightface onnxruntime pyserial opencv-python-headless numpy --break-system-packages

CMD ["/bin/bash"]