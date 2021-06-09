import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os
import numpy as np
import scipy.io
import torch

def visualize_gesture_predictions(out_dir, model_dir, exps_to_compare, path_to_colins_result, exp_descriptions,
                                  sequence_to_visualize, exps_to_evaluate, eval_scheme, eval_freq, model_no):
    # calculate average recognition accuracy for each video to determine which video (-->gesture sequence) to visualize
    metric = 'acc'
    eval_type = 'plain'
    avg_exp_results = {}
    for exp in exps_to_evaluate:
        eval_file = os.path.join(model_dir, "Eval", eval_scheme, "{}Hz".format(eval_freq), eval_type, exp,
                                 "{}.pth.tar".format(model_no))
        eval_results = torch.load(eval_file)
        for key in eval_results:
            if key == 'overall':
                pass
            else:  # key = video_id
                results_per_video = eval_results[key]
                if key not in avg_exp_results:
                    avg_exp_results[key] = []
                avg_exp_results[key].append(results_per_video[metric])
    for video_id in avg_exp_results:
        avg_exp_results[video_id] = np.mean(avg_exp_results[video_id])
    avg_exp_results = [(video_id, avg_accuracy) for video_id, avg_accuracy in avg_exp_results.items()]
    avg_exp_results = sorted(avg_exp_results, key=lambda x: x[1])
    if sequence_to_visualize == "lowest":
        sequence = avg_exp_results[0][0]
    elif sequence_to_visualize == "highest":
        sequence = avg_exp_results[-1][0]
    elif sequence_to_visualize == "median":
        sequence = avg_exp_results[len(avg_exp_results) // 2][0]
    elif sequence_to_visualize == "median":
        sequence = avg_exp_results[len(avg_exp_results) // 2][0]                 
    else:
      print("Unclear which sequence to visualize. Should be one of ['lowest', 'median', 'highest']")
      return
    sequences_to_plot = []
    for exp in exps_to_compare:
        eval_file = os.path.join(model_dir, "Eval", eval_scheme, "{}Hz".format(eval_freq), exp,
                                 "{}.pth.tar".format(model_no))
        eval_results = torch.load(eval_file)
        sequences_to_plot.append(eval_results[sequence]['P'])
    if path_to_colins_result:  # find results reproduced from Colin Lea et al.
        data_splits = {'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}
        sequence_id = sequence.split('_')[-1]  # e.g. "F002"
        user_id = sequence_id[0]
        trial_no = int(sequence_id[1:])
        mat = scipy.io.loadmat(os.path.join(path_to_colins_result, "Split_{}.mat".format(data_splits[user_id])))
        split_results = mat['P'].squeeze()
        trial_result = split_results[trial_no - 1].squeeze()
        sequences_to_plot.append(trial_result)
        
        
