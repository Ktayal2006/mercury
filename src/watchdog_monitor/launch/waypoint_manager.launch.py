from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg = get_package_share_directory('watchdog_monitor')
    waypoints_yaml = os.path.join(pkg, 'config', 'waypoints.yaml')

    declare_odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/diff_drive_controller/odom',
        description='Odometry topic to use for pose estimation'
    )
    declare_radius_arg = DeclareLaunchArgument(
        'arrival_radius',
        default_value='0.5',
        description='Arrival radius in metres'
    )

    waypoint_detector_node = Node(
        package='watchdog_monitor',
        executable='waypoint_detector_node',
        name='waypoint_detector_node',
        output='screen',
        parameters=[
            waypoints_yaml,
            {
                'odom_topic': LaunchConfiguration('odom_topic'),
                'arrival_radius': LaunchConfiguration('arrival_radius'),
            }
        ]
    )

    return LaunchDescription([
        declare_odom_topic_arg,
        declare_radius_arg,
        waypoint_detector_node,
    ])
