from setuptools import setup

package_name = 'plannerV0'

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
            'talker = plannerV0.publisher_member_function:main',
            'listener = plannerV0.subscriber_member_function:main',
            'worldcom = plannerV0.worldcom:main',
            'dummyserver = plannerV0.dummyserver:main',
            'async_client = plannerV0.async_client:main',
        ],
    },
)
