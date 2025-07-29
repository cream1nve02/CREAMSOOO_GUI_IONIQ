#!/usr/bin/env python3

import subprocess
import sys
import os

# ROS 환경 설정
os.environ['ROS_MASTER_URI'] = 'http://localhost:11311' 
os.environ['ROS_HOSTNAME'] = 'localhost'

# 현재 디렉토리를 작업공간으로 설정
os.chdir('/home/inji/ioniq_ws')

print("🚗 Starting CREAMSOOO IONIQ GUI...")
print("🎯 High-performance PyQt5 + OpenCV GUI")
print("📺 Smooth 30fps real-time video display")
print("🎮 Full GUI controls and real-time data")
print("")

try:
    # PyQt GUI 실행
    subprocess.run([sys.executable, 'src/creamsooo_gui.py'])
except KeyboardInterrupt:
    print("\n👋 Native GUI stopped")
except Exception as e:
    print(f"Error: {e}")
    print("Requirements:")
    print("  pip3 install PyQt5 opencv-python")
    print("  sudo apt-get install python3-pyqt5")