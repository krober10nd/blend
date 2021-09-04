from setuptools import setup, find_packages

setup(
    name="blend",
    version="0.1.0",
    license="GPL v3",
    description="geophysical data fusion,
    author="Keith J. Roberts",
    url="https://github.com/krober10nd/blend",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
)
