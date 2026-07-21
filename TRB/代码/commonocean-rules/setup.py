from setuptools import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    name='commonocean-rules',
    version='1.0.3',
    description='Implementation of traffic rule monitor for ships',
    keywords='rule monitor for ships',
    url='https://commonocean.cps.cit.tum.de/',
    author='Cyber-Physical Systems Group, Technical University of Munich',
    author_email='commonocean@lists.lrz.de',
    license="GNU General Public License v3.0",
    packages=find_packages(exclude=['scripts']),
    long_description_content_type='text/markdown',
    long_description=readme,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    install_requires=[
        'commonocean-io >= 2025.1',
        'python-monitors >= 0.1.1',
        'scipy',
        'numpy >= 1.16.4',
        'metric-temporal-logic >= 0.1.4',
        'matplotlib >= 2.5.0',
        'ruamel.yaml >= 0.16.10',
        'bezier >= 2020.2.3',
        'antlr4-python3-runtime>=4.7.2',
    ],

    data_files=[('', ['LICENSE']), ('', ['rules/config_ship.yaml']), ('', ['rules/traffic_rules_ship.yaml'])],
    include_package_data=True,
)
