import glob
import copy
import os
import subprocess
import argparse

import numpy as np
import matplotlib.pyplot as plt

from commonocean.common.file_reader import CommonOceanFileReader
from commonocean.visualization.draw_dispatch_cr import draw_object

from rules.common.commonocean_evaluation_ship import CommonOceanObstacleEvaluation


VIS_EGO = {
    'draw_shape': True,
    'draw_icon': False,
    'draw_bounding_box': True,
    'trajectory_steps': 2,
    'show_label': False,
    'occupancy': {
        'draw_occupancies': 0, 'shape': {
            'rectangle': {
                'opacity': 0.2,
                'facecolor': '#00b300',  # bright green '#fa0200', # bright red
                'edgecolor': '#00b300',  # bright green '#0066cc', # default blue
                'linewidth': 0.5,
                'zorder': 18,
            }}},
    'shape': {'rectangle': {'opacity': 0.8,
                            'facecolor': '#00b300',  # bright green '#fa0200', # bright red
                            'edgecolor': '#00b300',  # bright green '#831d20', # dark red
                            'linewidth': 0.5,
                            'zorder': 20}

              }}
VIS_RULE_VIOLATION = {
    'draw_shape': True,
    'draw_icon': False,
    'draw_bounding_box': True,
    'trajectory_steps': 2,
    'show_label': False,
    'occupancy': {
        'draw_occupancies': 0, 'shape': {
            'rectangle': {
                'opacity': 0.2,
                'facecolor': '#fa0200',  # bright red '#00b300', # bright green
                'edgecolor': '#fa0200',  # bright red '#0066cc', # default blue
                'linewidth': 0.5,
                'zorder': 18,
            }}},
    'shape': {'rectangle': {'opacity': 0.8,
                            'facecolor': '#fa0200',  # bright red '#00b300', # bright green
                            'edgecolor': '#fa0200',  # bright red '#831d20', # dark red
                            'linewidth': 0.5,
                            'zorder': 20}

              }}


def get_args():

    parser = argparse.ArgumentParser(description="Traffic Rule Evaluation Visualization of CommonOcean scenarios")
    parser.add_argument('--output_path', default='rules/output/viz/', type=str, help='Path to output directory')
    parser.add_argument('--input_path', default='scenarios/evaluation/05_21_MarineCadastre/Florida/', type=str, help='Path to directory with scenario files')
    parser.add_argument('--scenario', default='USA_FLO-1_20190101_T-1', type=str, help='Benchmark id of scenario to be visualized')
    parser.add_argument('--activated_rule_set', nargs='+',type=str, help='Activated traffic rules')

    return parser.parse_args()


def render(scenario, current_timestep, plot_limits, ego_vehicle, rule_violation,  output_path):
    """
    plotting the scenario for a given time range
    """

    if rule_violation:
        ego_viz_params = VIS_RULE_VIOLATION
    else:
        ego_viz_params = VIS_EGO

    i = current_timestep
    file_name_format = scenario.benchmark_id + '_timestep_%03d.png'
    plt.clf()
    plt.cla()
    plt.close()
    plt.figure(figsize=(20, 20))
    plt.gca().axis("equal")
    draw_object(scenario, plot_limits=plot_limits, draw_params={'time_begin': i,'trajectory_steps': 0})
    draw_object(ego_vehicle, draw_params={'time_begin': i, 'dynamic_obstacle': ego_viz_params})
    plt.savefig(output_path + file_name_format % i, bbox_inches='tight', dpi=600)
    pass


def create_video(benchmark_id, ego_id, output_path):
    capture_frame_rate = 10
    file_name_format_begin = benchmark_id
    file_name_format_end = '_timestep_%03d.png'
    for video_cnt in range(1):
        file_name_format = file_name_format_begin + file_name_format_end
        video_file_name = file_name_format_begin + '_' + str(ego_id) + '.mp4'
        os.chdir(output_path)
        subprocess.call(
            ['ffmpeg', '-r', str(capture_frame_rate), '-i', str(file_name_format), '-c:v', 'libx264', '-vf',
             'fps=25', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white', '-pix_fmt', 'yuv420p', str(video_file_name)])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        os.chdir(os.path.dirname(os.getcwd()))
    pass


def get_max_time(scenario):
    max_time = 0
    for obs in scenario.dynamic_obstacles:
        if obs.prediction is None:
            continue
        if obs.prediction.trajectory.state_list[-1].time_step > max_time:
            max_time = obs.prediction.trajectory.state_list[-1].time_step
    return max_time


def main():

    # open scenario
    args = get_args()
    scenario, planning_problem_set = CommonOceanFileReader(args.input_path + args.scenario).open()

    # generate rule evaluation
    cr_eval = CommonOceanObstacleEvaluation(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/rules/", args.activated_rule_set)
    cr_eval.simulation_param["generate_viz"] = True
    cr_eval.simulation_param["operating_mode"] = "single_scenario"
    print("Evaluating the scenario ... ")
    result = cr_eval.evaluate_scenario(scenario, flag_print = False) # single scenario mode for selected scenario with all rules
    print("Done!")
    print("Generating video ... (this may take a while)")

    # get plot limits
    min_x = min_y =  100000000000
    max_x = max_y = - 100000000000
    for obstacle in scenario.dynamic_obstacles:
        initial_state = obstacle.initial_state.position
        final_state = obstacle.prediction.trajectory.state_list[-1].position
        min_x = min(min_x, min(initial_state[0], final_state[0]))
        min_y = min(min_y, min(initial_state[1], final_state[1]))
        max_x = max(max_x, max(initial_state[0], final_state[0]))
        max_y = max(max_y, max(initial_state[1], final_state[1]))

    plot_limits = [min_x, max_x, min_y, max_y]

    # iterate through dynamic obstacles
    if not scenario.dynamic_obstacles:
        print("Video was not generated since the Scenario has no dynamic obstacles!")
    if len(scenario.dynamic_obstacles) < 2:
        print("You need at least 2 dynamic obstacles in the scenario to generate the video!")
    else:
        for obstacle in scenario.dynamic_obstacles:
            ego_id = obstacle.obstacle_id
            scenario_wo_ego = copy.deepcopy(scenario)
            scenario_wo_ego.remove_obstacle(obstacle)
            dict_rules = {}
            for eval in result:
                if eval[0] == ego_id:
                    dict_rules = eval[1]
            rule_violation = [False] * get_max_time(scenario)
            for key, value in dict_rules.items():
                if value[0] != -1:
                    if len(value) == 2:
                        rule_violation_temp = [True] * int((value[1] - value[0]) / scenario.dt)
                        if len(rule_violation_temp) < len(rule_violation):
                            if value[0] != 0:
                                temp = [False] * ((value[0]) / scenario.dt)
                                temp.extend(rule_violation_temp)
                                rule_violation_temp = temp
                            if value[1] != len(rule_violation) - 1 and len(rule_violation_temp) < len(rule_violation):
                                temp = [False] * int((len(rule_violation)) - (value[1] / scenario.dt))
                                rule_violation_temp.extend(temp)
                        rule_violation = np.any([rule_violation_temp, rule_violation], axis=0)
                    else:
                        print('Error: more than 2 switches between true and false not implemented!')

            # render
            for i in range(0, get_max_time(scenario)):
                render(scenario_wo_ego, i, plot_limits, obstacle, rule_violation[i], args.output_path)

            # create and store video
            create_video(scenario.benchmark_id, ego_id, args.output_path)


if __name__ == "__main__":
    main()
