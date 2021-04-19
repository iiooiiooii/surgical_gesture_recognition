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

                 
                    
                    
