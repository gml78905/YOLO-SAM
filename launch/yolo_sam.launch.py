#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("yolo_sam"), "config", "yolo_sam.yaml"]
                ),
                description="Path to yolo_sam ROS parameters YAML",
            ),
            Node(
                package="yolo_sam",
                executable="yolo_sam_node",
                name="yolo_sam_node",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
