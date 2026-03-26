#!/usr/bin/env python3
"""
face_task.launch.py
===================
Launch all three face recognition nodes together.

Usage:
    ros2 launch perception face_task.launch.py \
        target_image_path:=/path/to/target.jpg \
        dry_run:=false \
        serial_port:=/dev/ttyUSB1 \
        similarity_threshold:=0.35

To start the task after launching:
    ros2 topic pub /face_task/start std_msgs/Bool "data: true" --once

To monitor:
    ros2 topic echo /face/best_similarity
    ros2 topic echo /face_task/complete
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        # ── Launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument('target_image_path',    default_value='',
            description='Absolute path to target face image (JPG/PNG)'),

        DeclareLaunchArgument('similarity_threshold', default_value='0.35',
            description='ArcFace cosine similarity threshold (0.30-0.40 recommended)'),

        DeclareLaunchArgument('detection_size',       default_value='320',
            description='InsightFace detection input size (320 or 640)'),

        DeclareLaunchArgument('serial_port',          default_value='/dev/ttyUSB1',
            description='Arduino serial port'),

        DeclareLaunchArgument('dry_run',              default_value='true',
            description='dry_run=true: log commands only, no real serial/hardware'),

        DeclareLaunchArgument('settle_time_sec',      default_value='0.5',
            description='Seconds to wait after moving turret before capturing'),

        DeclareLaunchArgument('laser_on_time_sec',    default_value='3.0',
            description='Duration to keep laser on after match'),

        # ── Node 1: Vision ─────────────────────────────────────────────────────
        Node(
            package='perception',
            executable='face_recognition',
            name='face_recognition_node',
            output='screen',
            parameters=[{
                'target_image_path':    LaunchConfiguration('target_image_path'),
                'similarity_threshold': LaunchConfiguration('similarity_threshold'),
                'detection_size':       LaunchConfiguration('detection_size'),
            }]
        ),

        # ── Node 2: Task brain ─────────────────────────────────────────────────
        Node(
            package='perception',
            executable='face_task',
            name='face_task_node',
            output='screen',
            parameters=[{
                'settle_time_sec':   LaunchConfiguration('settle_time_sec'),
                'laser_on_time_sec': LaunchConfiguration('laser_on_time_sec'),
            }]
        ),

        # ── Node 3: Hardware bridge ────────────────────────────────────────────
        Node(
            package='perception',
            executable='turret_controller',
            name='turret_controller_node',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'dry_run':     LaunchConfiguration('dry_run'),
            }]
        ),

    ])
