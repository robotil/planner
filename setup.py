from setuptools import setup

package_name = 'v0_planner'

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
    maintainer='robil',
    maintainer_email='mhallakstamler@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = v0_planner.publisher_member_function:main',
            'listener = v0_planner.subscriber_member_function:main',
            'worldcom = v0_planner.worldcom:main',
            'dummyserver = v0_planner.dummyserver:main',
            'async_client = v0_planner.async_client:main',
        ],
    },
)
