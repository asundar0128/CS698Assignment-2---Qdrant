from setuptools import setup

package_name = 'camera_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Camera Publisher Node',
    entry_points={
        'console_scripts': [
            'camera_publisher = camera_publisher.camera_publisher_node:main'
        ],
    },
)
