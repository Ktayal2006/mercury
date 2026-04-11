from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    declare_interval_arg = DeclareLaunchArgument(
        'monitor_interval',
        default_value='2.0',
        description='Seconds between each health check cycle'
    )

    system_monitor_node = Node(
        package='watchdog_monitor',
        executable='system_monitor_node',
        name='system_monitor_node',
        output='screen',
        parameters=[{
            'monitor_interval': LaunchConfiguration('monitor_interval'),
            'expected_nodes': [
                '/robot_state_publisher',
                '/ekf_filter_node',
                '/slam_toolbox',
                '/lifecycle_manager_localization',
                '/planner_server',
                '/controller_server',
                '/bt_navigator',
                '/behavior_server',
                '/lifecycle_manager_navigation',
                '/diff_drive_controller',
                '/joint_state_broadcaster',
                '/lane_costmap',
            ],
        }]
    )

    return LaunchDescription([
        declare_interval_arg,
        system_monitor_node,
    ])
