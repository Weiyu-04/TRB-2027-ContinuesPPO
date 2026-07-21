# CommonOcean Rules

This repository contains a python traffic rule monitor for ships. The implemented rules and parameters are described
in the paper "Temporal Logic Formalization of Marine Traffic Rules" by Hanna Krasowski and Matthias Althoff.

## Getting Started
This introduction will give you an overview how to install, parametrize, 
and execute the Marine Traffic Rule Monitor. The scenarios to evaluate have to be represented in the [CommonOcean format](https://commonocean.cps.cit.tum.de).

We recommend Ubuntu as operating system.
For the monitor you need at least Python 3.8 and the following packages:

* *commonocean-io* >= 2025.1
* *matplotlib* >= 2.5.0
* *numpy* >= 1.16.4
* *metric-temporal-logic* >= 0.1.4
* *python-monitors* >= 0.1.1
* *ruamel.yaml* >= 0.16.10 
* *bezier* >= 2020.2.3
* *antlr4-python3-runtime* >= 4.7.2
* *scipy*

The usage of the Anaconda Python distribution is recommended.

If you want to use our functions and classes in your custom projects (e.g., in the [CommonOcean Drivability Checker](https://commonocean.cps.cit.tum.de/commonocean-dc)), you can install CommonOcean Rules as a module by doing
```
pip install commonocean-rules
```

If you want to create scenario videos, the ffmpeg packages has to be installed as well:
```
sudo snap install ffmpeg
```

Or, if you are a macOS user:
```
brew install ffmpeg
```

## Running the Marine Traffic Rule Monitor
There exist two options to run the Marine Traffic Rule Monitor:
* *[main.py](./rules/main.py)* : Traffic rule evaluation for a single scenario or all scenarios which are located within predefined directories.
* *[solution_viz.py](./rules/solution_viz.py)* : Traffic rule evaluation for a single scenario while also generating a video for the scenario.

The main file can be started from the root directory of this project via commandline by executing:
```
python -m rules.main --max_num_scenarios #Scenarios --scenario_directories /directory_1/ ... /directory_2
```

This should be used for the sequential evaluation of scenarios.
The scenario directories are relative to the root folder, remember to keep them inside the repo.

In the following the different parameters are explained:
* **max_num_scenarios**: The maximum number scenarios which should be evaluated. The default number is *10*.
* **scenario_directories**: List of scenario directories (relative to the root directory).

You can list the different parameters by executing `python main.py -h`. 

The temporal logic formulas of the rules and the rule specific parametrization can be found in the file *[traffic_rules_ship.yaml](./rules/traffic_rules_ship.yaml)*.  
Simulation and vehicle parameters can be set in the file *[config_ship.yaml](./rules/config_ship.yaml)*.

In adition, the solution_viz file can be started from the root directory of the project via commandline by executing
```
python -m rules.solution_viz --input_path /folder/of/input/scenario/ --scenario 'ScenarioFile.xml' --output_path /folder/for/output/video/
```

Pay attention that here the folders' paths need to be the absolute ones, as they represent the directories in which we get our scenario and then output the video. Also, don't forget the / (forward slash) in the end of the path name.

## Changelog

Compared to version 1.0.0, the following features have been changed:

- The module now uses the new version of CommonOcean IO (2023.1)
- The package is no longer compatible with Python 3.7

## Authors of code project

Hanna Krasowski, Sebastian Maierhofer, Bruno Maione

**If you use our formalized traffic rules for research, please consider citing our paper:**
```
@inproceedings{Krasowski2021a,
	author = {Krasowski, Hanna and Althoff, Matthias},
	title = {Temporal Logic Formalization of Marine Traffic Rules},
	booktitle = {Proc. of the IEEE Intelligent Vehicles Symposium},
	year = {2021},
}
```


