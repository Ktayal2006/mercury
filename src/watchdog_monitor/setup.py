from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'watchdog_monitor'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description='Combined monitoring package: system_monitor, watchdog, waypoint_manager, control_listener',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # system_monitor
            'system_monitor_node = watchdog_monitor.system_monitor_node:main',
            'monitoring_dashboard = watchdog_monitor.monitoring_dashboard:main',
            # watchdog
            'watchdog_node = watchdog_monitor.watchdog_node:main',
            # waypoint_manager
            'waypoint_detector_node = watchdog_monitor.waypoint_detector_node:main',
            # control_listener
            'control_listener_node = watchdog_monitor.control_listener_node:main',
        ],
    },
)
