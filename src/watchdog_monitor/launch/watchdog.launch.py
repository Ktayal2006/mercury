from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    declare_check_interval_arg = DeclareLaunchArgument(
        'check_interval',
        default_value='3.0',
        description='Seconds between each watchdog check cycle'
    )
    declare_topic_timeout_arg = DeclareLaunchArgument(
        'topic_timeout',
        default_value='5.0',
        description='Seconds of silence before a topic is flagged as inactive'
    )

    watchdog_node = Node(
        package='watchdog_monitor',
        executable='watchdog_node',
        name='watchdog_node',
        output='screen',
        parameters=[{
            'check_interval': LaunchConfiguration('check_interval'),
            'topic_timeout': LaunchConfiguration('topic_timeout'),
            'critical_nodes': [
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
            'tf_pairs': [
                'odom->base_link',
                'map->odom',
                'base_link->laser',
            ],
        }]
    )

    return LaunchDescription([
        declare_check_interval_arg,
        declare_topic_timeout_arg,
        watchdog_node,
    ])
