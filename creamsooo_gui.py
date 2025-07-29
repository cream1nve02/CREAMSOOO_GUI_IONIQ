#!/usr/bin/env python3

import sys, os, cv2, numpy as np, rospy, subprocess, re
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QGridLayout, QGroupBox, QPushButton,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QFileDialog, QSlider, QTextEdit, QSplitter, QButtonGroup, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# ROS imports
from sensor_msgs.msg import Image, CompressedImage
from lane_detection_ros.msg import LaneDetection
from std_msgs.msg import Float32, Bool, String, Int32 as ModeCommand
from cv_bridge import CvBridge
import roslib.message 

# RViz Python binding import
from rviz import bindings as rviz

# 선택적 임포트 처리
try:
    from vehicle_control.msg import Actuator
    VEHICLE_CONTROL_AVAILABLE = True
except ImportError:
    print("Warning: 'vehicle_control' ROS package not found.")
    VEHICLE_CONTROL_AVAILABLE = False
    class Actuator: pass

class ROSDataThread(QThread):
    image_received = pyqtSignal(str, np.ndarray)
    lane_data_received = pyqtSignal(LaneDetection)
    actuator_received = pyqtSignal(Actuator)
    steer_received = pyqtSignal(Float32)
    rl_velocity_received = pyqtSignal(Float32)
    rr_velocity_received = pyqtSignal(Float32)
    lidar_warn_received = pyqtSignal(Bool)
    vision_warn_received = pyqtSignal(Bool)
    lc_status_received = pyqtSignal(String)
    preview_message_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.is_running = True
        self.preview_subscriber = None

    def run(self):
        rospy.Subscriber("/gmsl_camera/dev/video4/compressed", CompressedImage, lambda msg: self.image_callback(msg, 'camera'), queue_size=1)
        rospy.Subscriber("/lane_detection/visualization", Image, lambda msg: self.image_callback(msg, 'lane'), queue_size=1)
        rospy.Subscriber("/lane_detection", LaneDetection, lambda msg: self.lane_data_received.emit(msg), queue_size=1)
        if VEHICLE_CONTROL_AVAILABLE:
            rospy.Subscriber('/target_actuator', Actuator, lambda msg: self.actuator_received.emit(msg), queue_size=1)
            rospy.Subscriber('/vehicle/steering_angle', Float32, lambda msg: self.steer_received.emit(msg), queue_size=1)
            rospy.Subscriber('/vehicle/velocity_RL', Float32, lambda msg: self.rl_velocity_received.emit(msg), queue_size=1)
            rospy.Subscriber('/vehicle/velocity_RR', Float32, lambda msg: self.rr_velocity_received.emit(msg), queue_size=1)
            rospy.Subscriber('/mobinha/hazard_warning', Bool, lambda msg: self.lidar_warn_received.emit(msg), queue_size=1)
            rospy.Subscriber('/mobinha/is_crossroad', Bool, lambda msg: self.vision_warn_received.emit(msg), queue_size=1)
            rospy.Subscriber('/lane_change_status', String, lambda msg: self.lc_status_received.emit(msg), queue_size=1)
        rospy.spin()

    def image_callback(self, msg, topic_name):
        if not self.is_running: return
        try:
            if isinstance(msg, CompressedImage):
                cv_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_image is not None: self.image_received.emit(topic_name, cv_image)
        except Exception as e:
            print(f"Error processing {topic_name} image: {e}")
            
    def preview_callback(self, msg):
        self.preview_message_received.emit(str(msg))

    def change_preview_subscription(self, topic_name, topic_type):
        if self.preview_subscriber: self.preview_subscriber.unregister()
        try:
            msg_class = roslib.message.get_message_class(topic_type)
            if msg_class:
                self.preview_subscriber = rospy.Subscriber(topic_name, msg_class, self.preview_callback)
        except Exception as e:
            print(f"Failed to subscribe to preview topic {topic_name}: {e}")

    def stop(self):
        self.is_running = False

class TopicStatusThread(QThread):
    status_updated = pyqtSignal(dict)
    def __init__(self): super().__init__(); self.is_running = True
    def run(self):
        while self.is_running:
            try:
                active_topics = {name: type_str for name, type_str in rospy.get_published_topics()}
                self.status_updated.emit(active_topics)
            except Exception as e: print(f"Failed to get topic status: {e}")
            self.msleep(1000)
    def stop(self): self.is_running = False

class ImageDisplayWidget(QWidget):
    def __init__(self, title=""):
        super().__init__()
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title, font=QFont("Arial", 14, QFont.Bold), alignment=Qt.AlignCenter)
        self.image_label = QLabel("No Image", alignment=Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #555; background-color: black;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)
        self.image_mutex = QMutex()
        
    def update_image(self, cv_image):
        self.image_mutex.lock()
        try:
            h, w, ch = cv_image.shape
            qt_image = QImage(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        finally:
            self.image_mutex.unlock()


class VehicleStatusWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.target_accel, self.target_steer, self.target_brake, self.target_waypoint = 0.0, 0.0, 0.0, 0
        self.real_steer, self.rl_v, self.rr_v = 0.0, 0.0, 0.0
        self.lidar_warn, self.vision_warn = False, False
        self.lc_status, self.lc_rejected_time = "N/A", None
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("QLabel { color: white; font-size: 16pt; }")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 10, 15, 10)
        title = QLabel("📊 Vehicle Status", font=QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #55aaff; padding-bottom: 5px;")
        main_layout.addWidget(title)
        grid = QGridLayout()
        grid.setVerticalSpacing(4)
        self.labels = {}
        info_list = ["Driving Mode", "Detected Lanes", "Driving Lane", "Velocity [km/h]", "Target Accel [%]", "Target Brake [%]", "Target Steer [deg]", "Real Steer [deg]", "Obstacle", "Traffic Sign", "Lane Change"]
        row, col = 0, 0
        for text in info_list:
            label_title = QLabel(f"{text}:", font=QFont("Arial", 14))
            grid.addWidget(label_title, row, col)
            self.labels[text] = QLabel("N/A", font=QFont("Arial", 14, QFont.Bold))
            grid.addWidget(self.labels[text], row, col + 1)
            row += 1
            if row > 5: row, col = 0, 2
        grid.setColumnStretch(1, 1); grid.setColumnStretch(3, 1)
        main_layout.addLayout(grid)
        main_layout.addStretch()

    def update_all_displays(self):
        curr_v_kmh = (self.rl_v + self.rr_v) / 2 * 3.6
        mode_text = "Intersection" if self.target_waypoint == 0 else "Lane Following"
        self.labels["Driving Mode"].setText(f"<font color='#a29bfe'>{mode_text}</font>")
        self.labels["Target Accel [%]"].setText(f"{self.target_accel:.2f}")
        self.labels["Target Steer [deg]"].setText(f"{self.target_steer * 12:.2f}")
        self.labels["Real Steer [deg]"].setText(f"{self.real_steer:.2f}")
        self.labels["Target Brake [%]"].setText(f"{self.target_brake:.2f}")
        self.labels["Velocity [km/h]"].setText(f"{curr_v_kmh:.2f}")
        obs_text, obs_color = ("Detected", "#e74c3c") if self.lidar_warn else ("Clear", "#2ecc71")
        self.labels["Obstacle"].setText(f"<font color='{obs_color}'>{obs_text}</font>")
        sign_text, sign_color = ("Detected", "#e74c3c") if self.vision_warn else ("Clear", "#2ecc71")
        self.labels["Traffic Sign"].setText(f"<font color='{sign_color}'>{sign_text}</font>")
        lc_color = "#e67e22" if "command" in self.lc_status else ("#e74c3c" if "rejected" in self.lc_status else "white")
        self.labels["Lane Change"].setText(f"<font color='{lc_color}'>{self.lc_status}</font>")

    def update_lane_data(self, msg):
        lane_names = ["LL", "L", "R", "RR"]
        detected = [lane_names[i] for i, lane in enumerate(msg.lanes) if lane.exists]
        self.labels["Detected Lanes"].setText(f"<font color='#55ff7f'>{', '.join(detected) or 'None'}</font>")
        driving_text = "2차로" if len(msg.lanes) > 0 and msg.lanes[0].exists else "1차로"
        self.labels["Driving Lane"].setText(f"<font color='#f1c40f'>{driving_text}</font>")
        if not VEHICLE_CONTROL_AVAILABLE: self.update_all_displays()

    def update_actuator(self, msg): self.target_accel, self.target_steer, self.target_brake, self.target_waypoint = msg.accel, msg.steer, msg.brake, msg.is_waypoint; self.update_all_displays()
    def update_steer(self, msg): self.real_steer = msg.data; self.update_all_displays()
    def update_rl_v(self, msg): self.rl_v = msg.data; self.update_all_displays()
    def update_rr_v(self, msg): self.rr_v = msg.data; self.update_all_displays()
    def update_lidar(self, msg): self.lidar_warn = msg.data; self.update_all_displays()
    def update_vision(self, msg): self.vision_warn = msg.data; self.update_all_displays()
    def update_lc_status(self, msg):
        new_status = msg.data
        if new_status.startswith("Lanechange command"):
            if self.lc_status == "Lane change rejected" and self.lc_rejected_time and (rospy.Time.now() - self.lc_rejected_time).to_sec() < 1.0: return
            self.lc_status, self.lc_rejected_time = new_status, None
        elif new_status == "Lane change rejected": self.lc_status, self.lc_rejected_time = new_status, rospy.Time.now()
        else: self.lc_status, self.lc_rejected_time = new_status, None
        self.update_all_displays()

class MainDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processes, self.buttons = {}, {}
        self.button_styles = {"Camera": "#2980b9", "LiDAR": "#2980b9", "GPS": "#2980b9", "Lane Detection": "#16a085", "LiDAR Tracking": "#8e44ad", "LiDAR DL": "#8e44ad", "Integration": "#8e44ad", "CAN": "#d35400", "Auto Mode": "#d35400", "Control Logic": "#d35400"}
        self.topic_map = {"Camera": "/gmsl_camera/dev/video4/compressed", "LiDAR": "/lidar_points", "GPS": "/novatel/oem7/inspva"}
        self.selected_bag_path, self.bag_duration = None, 0
        self.topic_types = {}
        if VEHICLE_CONTROL_AVAILABLE:
            self.mode_cmd_pub = rospy.Publisher('/gui/mode_command', ModeCommand, queue_size=1)
        self.init_ui()
        self.start_ros_threads()
        self.status_timer = QTimer(self, timeout=self.check_process_status); self.status_timer.start(1000)
        self.bag_progress_timer = QTimer(self, timeout=self.update_bag_progress)

    def init_ui(self):
        self.setWindowTitle("Integrated Control & Monitoring Dashboard")
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.create_control_panel(), 1)
        display_tabs = QTabWidget()
        display_tabs.addTab(self.create_monitoring_grid(), "Live Monitoring")
        display_tabs.addTab(self.create_topic_status_tab(), "Topic Status")
        main_layout.addWidget(display_tabs, 3)
        self.statusBar().showMessage("Ready.")

    def create_control_panel(self):
        control_panel_group = QGroupBox("🚀 Control Panel", font=QFont("Arial", 14, QFont.Bold))
        main_vbox = QVBoxLayout()
        main_vbox.addWidget(self.create_rosbag_groupbox())
        main_vbox.addWidget(self.create_sensor_groupbox())
        main_vbox.addWidget(self.create_perception_groupbox())
        main_vbox.addWidget(self.create_vehicle_control_groupbox())
        main_vbox.addStretch()
        stop_all_button = QPushButton("🚨 EMERGENCY STOP ALL 🚨", font=QFont("Arial", 10, QFont.Bold), clicked=self.stop_all_processes)
        stop_all_button.setStyleSheet("background-color: #c0392b; color: white; padding: 8px; border-radius: 4px;")
        main_vbox.addWidget(stop_all_button)
        control_panel_group.setLayout(main_vbox)
        return control_panel_group
    
    def add_button(self, layout, name, command, cwd=None, use_shell=False):
        button = QPushButton(name, font=QFont("Arial", 10, QFont.Bold))
        button.clicked.connect(lambda _, n=name, cmd=command, d=cwd, s=use_shell: self.toggle_process(n, cmd, d, s))
        button.setProperty("default_color", self.button_styles.get(name, "#34495e"))
        layout.addWidget(button)
        self.buttons[name] = button

    def create_sensor_groupbox(self):
        groupbox = QGroupBox("Sensors")
        layout = QVBoxLayout(groupbox)
        self.add_button(layout, "Camera", ["sshpass", "-p", "cvlab525", "ssh", "-X", "cvlab@192.168.101.4", "camera"])
        self.add_button(layout, "LiDAR", ["sshpass", "-p", "cvlab525", "ssh", "-X", "inha@192.168.101.1", "lidar"])
        self.add_button(layout, "GPS", [os.path.expanduser("~/run_novatel.sh")])
        return groupbox

    def create_perception_groupbox(self):
        groupbox = QGroupBox("Perception")
        layout = QVBoxLayout(groupbox)
        self.add_button(layout, "Lane Detection", ["roslaunch", "lane_detection_ros", "libtorch_lane_detection.launch"])
        self.add_button(layout, "LiDAR Tracking", ["sshpass", "-p", "cvlab525", "ssh", "-X", "inha@192.168.101.1", "tracking"])
        self.add_button(layout, "LiDAR DL", ["sshpass", "-p", "cvlab525", "ssh", "-X", "inha@192.168.101.1", "./start_dl.sh"])
        self.add_button(layout, "Integration", ["rosrun", "lidar_tracking", "integration.py"])
        return groupbox
        
    def create_vehicle_control_groupbox(self):
        groupbox = QGroupBox("Vehicle Control")
        layout = QVBoxLayout(groupbox)
        self.add_button(layout, "CAN", "echo 'cvlab525' | sudo -S ./can.sh", cwd="/home/inha/creamsooo_ws/src/vehicle_control", use_shell=True)
        self.add_button(layout, "Auto Mode", ["python3", "can2morai.py"], cwd="/home/inha/creamsooo_ws/src/vehicle_control/start_code")
        self.add_button(layout, "Control Logic", ["roslaunch", "vehicle_control", "setting.launch"])
        mode_cmd_group = QGroupBox("Mode Commands")
        mode_cmd_layout = QGridLayout(mode_cmd_group)
        self.mode_status_label = QLabel("N/A", font=QFont("Arial", 12, QFont.Bold)); self.mode_status_label.setStyleSheet("color: #f1c40f;")
        go_btn = QPushButton("All Go (77)"); go_btn.clicked.connect(lambda: self.send_mode_command(77, "All Systems Go"))
        stop_btn = QPushButton("Stop (55)"); stop_btn.clicked.connect(lambda: self.send_mode_command(55, "Stop"))
        hard_stop_btn = QPushButton("Hard Stop (88)"); hard_stop_btn.clicked.connect(lambda: self.send_mode_command(88, "Hard Stop"))
        kill_btn = QPushButton("Kill (66)"); kill_btn.clicked.connect(lambda: self.send_mode_command(66, "Vehicle Killed"))
        mode_cmd_layout.addWidget(QLabel("Current Mode:"), 0, 0, 1, 2); mode_cmd_layout.addWidget(self.mode_status_label, 1, 0, 1, 2)
        mode_cmd_layout.addWidget(go_btn, 2, 0); mode_cmd_layout.addWidget(stop_btn, 2, 1)
        mode_cmd_layout.addWidget(hard_stop_btn, 3, 0); mode_cmd_layout.addWidget(kill_btn, 3, 1)
        layout.addWidget(mode_cmd_group)
        course_group = QGroupBox("Course Selection")
        course_layout = QHBoxLayout(course_group)
        self.course_button_group = QButtonGroup(self)
        course_a = QPushButton("Course A"); course_a.setCheckable(True)
        course_b = QPushButton("Course B"); course_b.setCheckable(True)
        course_c = QPushButton("Course C"); course_c.setCheckable(True)
        self.course_button_group.addButton(course_a); self.course_button_group.addButton(course_b); self.course_button_group.addButton(course_c)
        course_a.setChecked(True)
        course_layout.addWidget(course_a); course_layout.addWidget(course_b); course_layout.addWidget(course_c)
        layout.addWidget(course_group)
        return groupbox
        
    def create_rosbag_groupbox(self):
        bag_group = QGroupBox("ROS Bag Control")
        layout = QVBoxLayout()
        file_layout = QHBoxLayout(); self.bag_path_label = QLabel("No Bag File Selected"); browse_btn = QPushButton("Browse", clicked=self.browse_bag_file); file_layout.addWidget(self.bag_path_label, 1); file_layout.addWidget(browse_btn); layout.addLayout(file_layout)
        self.bag_slider = QSlider(Qt.Horizontal, sliderReleased=self.seek_bag); self.bag_slider.setEnabled(False); layout.addWidget(self.bag_slider)
        time_layout = QHBoxLayout(); self.current_time_label = QLabel("00:00"); self.total_time_label = QLabel("00:00"); time_layout.addWidget(self.current_time_label); time_layout.addStretch(); time_layout.addWidget(self.total_time_label); layout.addLayout(time_layout)
        playback_layout = QHBoxLayout(); self.play_pause_bag_btn = QPushButton("▶️ Play", clicked=self.toggle_play_pause_bag); self.play_pause_bag_btn.setEnabled(False); self.stop_bag_btn = QPushButton("⏹️ Stop", clicked=self.stop_bag); playback_layout.addWidget(self.play_pause_bag_btn); playback_layout.addWidget(self.stop_bag_btn); layout.addLayout(playback_layout)
        bag_group.setLayout(layout)
        return bag_group

    def create_monitoring_grid(self):
        monitoring_widget = QWidget()
        main_layout = QHBoxLayout(monitoring_widget)
        self.rviz_frame = self.create_rviz_panel()
        
        # ❗❗❗ RViz와 우측 패널의 비율을 2:3으로 수정 ❗❗❗
        main_layout.addWidget(self.rviz_frame, 5) 
        
        right_side_layout = QVBoxLayout()
        self.camera_display = ImageDisplayWidget("📹 Camera")
        self.lane_display = ImageDisplayWidget("🛣️ Lane Detection Image")
        self.vehicle_status_widget = VehicleStatusWidget()
        
        right_side_layout.addWidget(self.camera_display, 3)
        right_side_layout.addWidget(self.lane_display, 3)
        right_side_layout.addWidget(self.vehicle_status_widget, 2)
        
        main_layout.addLayout(right_side_layout, 2)
        
        return monitoring_widget

    def create_rviz_panel(self):
        frame = rviz.VisualizationFrame()
        frame.setSplashPath("")
        frame.initialize()
        
        # RViz 패널 크기 제한 설정
        frame.setMaximumWidth(800)  # 최대 폭을 800픽셀로 제한
        frame.setMinimumWidth(600)  # 최소 폭을 600픽셀로 설정
        
        rviz_config_path = "/home/inha/creamsooo_ws/src/LiDAR-Tracking/integration/integration_casey.rviz"
        if os.path.exists(rviz_config_path):
            frame.load(rviz.Config().readFile(rviz_config_path))
        else:
            manager = frame.getManager(); manager.setFixedFrame("base_link"); manager.createDisplay("rviz/Grid", "Default Grid", True)
        return frame

    def create_topic_status_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        splitter = QSplitter(Qt.Vertical)
        self.topic_table = QTableWidget(columnCount=2)
        self.topic_table.setHorizontalHeaderLabels(["Topic Name", "Type"])
        self.topic_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.topic_table.setStyleSheet("QTableWidget { background-color: #34495e; gridline-color: #7f8c8d; }")
        self.topic_table.cellClicked.connect(self.on_topic_clicked)
        self.echo_preview = QTextEdit(readOnly=True)
        self.echo_preview.setStyleSheet("QTextEdit { background-color: black; color: lightgreen; font-family: 'Monospace'; }")
        splitter.addWidget(self.topic_table); splitter.addWidget(self.echo_preview)
        splitter.setSizes([400, 200])
        layout.addWidget(splitter)
        return widget

    def start_ros_threads(self):
        self.ros_thread = ROSDataThread()
        self.ros_thread.image_received.connect(self.update_image)
        self.ros_thread.lane_data_received.connect(self.vehicle_status_widget.update_lane_data)
        self.ros_thread.preview_message_received.connect(self.update_echo_preview)
        if VEHICLE_CONTROL_AVAILABLE:
            self.ros_thread.actuator_received.connect(self.vehicle_status_widget.update_actuator)
            self.ros_thread.steer_received.connect(self.vehicle_status_widget.update_steer)
            self.ros_thread.rl_velocity_received.connect(self.vehicle_status_widget.update_rl_v)
            self.ros_thread.rr_velocity_received.connect(self.vehicle_status_widget.update_rr_v)
            self.ros_thread.lidar_warn_received.connect(self.vehicle_status_widget.update_lidar)
            self.ros_thread.vision_warn_received.connect(self.vehicle_status_widget.update_vision)
            self.ros_thread.lc_status_received.connect(self.vehicle_status_widget.update_lc_status)
        self.ros_thread.start()
        self.topic_thread = TopicStatusThread()
        self.topic_thread.status_updated.connect(self.update_topic_status)
        self.topic_thread.start()
        
    def update_image(self, topic_name, cv_image):
        if topic_name == 'camera': self.camera_display.update_image(cv_image)
        elif topic_name == 'lane': self.lane_display.update_image(cv_image)

    def update_topic_status(self, active_topics):
        self.topic_types = active_topics
        self.topic_table.setRowCount(len(active_topics))
        for row, (name, type_str) in enumerate(sorted(active_topics.items())):
            self.topic_table.setItem(row, 0, QTableWidgetItem(name))
            self.topic_table.setItem(row, 1, QTableWidgetItem(type_str.split('/')[-1]))
            
    def on_topic_clicked(self, row, column):
        topic_name = self.topic_table.item(row, 0).text()
        topic_type = self.topic_types.get(topic_name)
        if topic_type:
            self.ros_thread.change_preview_subscription(topic_name, topic_type)
            self.echo_preview.setText(f"Subscribing to {topic_name}...")

    def update_echo_preview(self, message_text):
        self.echo_preview.setText(message_text)
    
    def check_process_status(self):
        active_topics = self.topic_types.keys()
        for name, button in self.buttons.items():
            is_process_running = name in self.processes and self.processes[name].poll() is None
            topic_to_check = self.topic_map.get(name)
            is_topic_active = topic_to_check and topic_to_check in active_topics
            is_on = is_process_running or is_topic_active
            color = "#27ae60" if is_on else button.property("default_color")
            text = f"{name} (working)" if is_topic_active else name
            button.setText(text)
            button.setStyleSheet(f"background-color: {color}; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
            
    def send_mode_command(self, cmd_id, status_text):
        if hasattr(self, 'mode_cmd_pub'):
            msg = ModeCommand(); msg.data = cmd_id
            self.mode_cmd_pub.publish(msg)
            self.mode_status_label.setText(status_text)
            self.statusBar().showMessage(f"Sent Mode Command: {status_text}", 3000)
        else:
            self.statusBar().showMessage("Cannot send command. 'vehicle_control' not available.", 3000)
            
    def toggle_process(self, name, command, cwd=None, use_shell=False):
        if name in self.processes and self.processes[name].poll() is None: self.stop_single_process(name)
        else: self.run_process(name, command, cwd, use_shell)

    def run_process(self, name, command, cwd=None, use_shell=False):
        try:
            self.statusBar().showMessage(f"Starting '{name}'...", 5000)
            proc = subprocess.Popen(command, cwd=cwd, shell=use_shell)
            self.processes[name] = proc
        except Exception as e:
            self.statusBar().showMessage(f"Failed to start '{name}': {e}", 5000)
            
    def stop_single_process(self, name):
        if name in self.processes:
            proc = self.processes[name]
            if proc.poll() is None:
                try: self.statusBar().showMessage(f"Stopping '{name}'...", 3000); proc.terminate(); proc.wait(timeout=2)
                except subprocess.TimeoutExpired: proc.kill()
            del self.processes[name]

    def stop_all_processes(self):
        self.statusBar().showMessage("Stopping all running processes...", 5000)
        self.stop_bag()
        for name in list(self.processes.keys()): self.stop_single_process(name)
        self.statusBar().showMessage("All processes have been stopped.", 3000)

    def browse_bag_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ROS Bag File", os.path.expanduser("~"), "Bag files (*.bag)")
        if not path: return
        self.stop_bag(); self.selected_bag_path = path
        self.bag_path_label.setText(os.path.basename(path))
        self.get_bag_info(path)
        
    def get_bag_info(self, path):
        try:
            info_str = subprocess.check_output(['rosbag', 'info', path]).decode('utf-8')
            duration_match = re.search(r"duration:\s*([\d\.]+)", info_str)
            if duration_match:
                self.bag_duration = int(float(duration_match.group(1)))
                self.bag_slider.setMaximum(self.bag_duration)
                self.total_time_label.setText(f"{self.bag_duration // 60:02d}:{self.bag_duration % 60:02d}")
                self.current_time_label.setText("00:00"); self.bag_slider.setValue(0)
                self.play_pause_bag_btn.setEnabled(True); self.bag_slider.setEnabled(True)
        except Exception as e:
            self.statusBar().showMessage(f"Error reading bag info: {e}", 5000)

    def is_bag_playing(self):
        return 'rosbag' in self.processes and self.processes['rosbag'].poll() is None

    def toggle_play_pause_bag(self):
        if not self.selected_bag_path: return
        if self.is_bag_playing(): self.stop_bag()
        else: self.play_bag()
            
    def play_bag(self):
        if not self.selected_bag_path or self.is_bag_playing(): return
        start_time = self.bag_slider.value()
        try:
            cmd = ['rosbag', 'play', self.selected_bag_path, '--start', str(start_time), '--clock', '--loop']
            proc = subprocess.Popen(cmd)
            self.processes['rosbag'] = proc
            self.bag_progress_timer.start(1000)
            self.play_pause_bag_btn.setText("⏸️ Pause")
        except Exception as e:
            self.statusBar().showMessage(f"Failed to play bag: {e}", 5000)

    def stop_bag(self):
        self.bag_progress_timer.stop()
        self.stop_single_process('rosbag')
        self.play_pause_bag_btn.setText("▶️ Play")
            
    def seek_bag(self):
        if not self.selected_bag_path: return
        seek_time = self.bag_slider.value()
        self.current_time_label.setText(f"{seek_time // 60:02d}:{seek_time % 60:02d}")
        if self.is_bag_playing():
            self.stop_bag(); self.play_bag()

    def update_bag_progress(self):
        if self.is_bag_playing():
            new_val = self.bag_slider.value() + 1
            if new_val > self.bag_duration: new_val = 0
            self.bag_slider.setSliderPosition(new_val) 
            self.current_time_label.setText(f"{new_val // 60:02d}:{new_val % 60:02d}")
        else:
            if self.bag_progress_timer.isActive(): self.stop_bag()
            
    def closeEvent(self, event):
        self.stop_all_processes()
        self.image_thread.stop(); self.topic_thread.stop()
        self.image_thread.quit(); self.topic_thread.quit()
        self.image_thread.wait(); self.topic_thread.wait()
        event.accept()

def main():
    try:
        rospy.init_node('integrated_dashboard_gui', anonymous=True)
    except rospy.ROSInterruptException:
        print("ROS node initialization failed."); sys.exit(1)
    app = QApplication(sys.argv)
    window = MainDashboard()
    window.resize(1600, 850)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()