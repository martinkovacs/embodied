from setuptools import setup

setup(
    name="arm_node",
    version="0.1.0",
    packages=["arm_node"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/arm_node"]),
                ("share/arm_node", ["package.xml"])],
    entry_points={"console_scripts": ["arm_node = arm_node.arm_node:main"]},
)
