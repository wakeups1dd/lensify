from setuptools import setup, find_packages

setup(
    name="lensify",
    version="0.2.0",
    description="Lensify upgraded local semantic search",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "lensify=lensify.cli:main",
        ]
    },
    install_requires=[],
)
