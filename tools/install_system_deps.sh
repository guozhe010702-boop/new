#!/bin/sh
set -eu
test "$(id -u)" = 0 || { echo 'Run as root'; exit 1; }
apt-get update
apt-get install -y --no-install-recommends build-essential cmake pkg-config \
  libopencv-dev libeigen3-dev libdrm-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good v4l-utils lsof x11-xserver-utils
