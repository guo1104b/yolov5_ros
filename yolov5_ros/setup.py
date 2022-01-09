from setuptools import setup

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wg',
    maintainer_email='wg@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tool_recognition_ros = yolov5_ros.tool_recognition_ros:main',
            'pub = yolov5_ros.pub:main',
        ],
    },
)
