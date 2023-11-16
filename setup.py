from setuptools import setup, find_packages

with open("requirements.txt") as f:
	install_requires = f.read().strip().split("\n")

# get version from __version__ variable in mybot/__init__.py
from mybot import __version__ as version

setup(
	name="mybot",
	version=version,
	description="Mybot",
	author="shuvo",
	author_email="shuvo.gmail.com",
	packages=find_packages(),
	zip_safe=False,
	include_package_data=True,
	install_requires=install_requires
)
