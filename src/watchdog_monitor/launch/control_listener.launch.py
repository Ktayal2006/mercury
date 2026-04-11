from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    control_listener_node = Node(
        package='watchdog_monitor',
        executable='control_listener_node',
        name='control_listener_node',
        output='screen',
    )

    return LaunchDescription([
        control_listener_node,
    ])
