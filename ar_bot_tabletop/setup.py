from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["ar_bot_tabletop"], package_dir={"": "scripts"}
)

setup(**setup_args)
