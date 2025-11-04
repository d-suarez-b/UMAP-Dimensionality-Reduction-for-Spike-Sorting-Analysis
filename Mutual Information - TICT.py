#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import shutil
from os.path import join as pjoin
from mpl_toolkits.mplot3d import axes3d
import matplotlib
from matplotlib import style
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import math
import pandas as pd
import numba
from numba import jit
from tqdm import tqdm
import time


def import_file(namefile, delimiter, start_column):
    """
    Imports data from a text file.

    Reads a file line by line, splits each line by the specified delimiter,
    and converts the elements from 'start_column' onwards into floats.
    It skips empty lines.

    Args:
        namefile (str): The path to the input file.
        delimiter (str): The delimiter string used to separate values.
        start_column (int): The 0-based index of the column from which 
                            to start reading data.

    Returns:
        list: A nested list where each inner list contains the
              float-converted data from a single line.
    """
    with open(namefile, 'r') as file:
        file_lines = file.readlines()
    
    data_list = []
    for line in file_lines:
        if line.strip():  # Avoid empty lines
            temp_list = line.strip().split(delimiter)
            data_list.append([float(i) for i in temp_list[start_column::]])
    return data_list

def firingRate_by_trial(spikes, t_start=-2, t_end=8, window_size=0.05, step_size=0.01, **kwargs):
    """
    Calculates firing rate per trial using a sliding window.

    This function processes a 2D spike array, where spikes are grouped
    by trial. It calculates the firing rate for each trial individually.
    Assumes `spikes` array format: [trial_id, spike_time, ..., class_id].

    Args:
        spikes (np.ndarray): A 2D array, [trial_id, spike_time, ..., class_id].
        t_start (float, optional): The start time for the analysis. Defaults to -2.
        t_end (float, optional): The end time for the analysis. Defaults to 8.
        window_size (float, optional): The duration of the sliding window. Defaults to 0.05.
        step_size (float, optional): The amount to slide the window forward. Defaults to 0.01.

    Returns:
        tuple:
            - np.ndarray: A 2D array (num_trials x num_bins) of firing rates.
            - np.ndarray: A 1D array (num_trials) of the class ID for each trial.
            - np.ndarray: A 1D array (num_bins) of the time vector.
    """
    trials = np.unique(spikes[:, 0])
    trial_classes = np.zeros(len(trials))
    
    num_bins = int(np.floor(((t_end - t_start) / step_size) - (window_size / step_size)) + 1)
    rate_matrix = np.zeros((len(trials), num_bins), dtype=np.float32)
    time_vector = np.zeros(num_bins)
    
    for i, trial_id in enumerate(trials):
        indices = np.where(spikes[:, 0] == trial_id)[0]
        col_index = 0
        current_time_window_start = t_start
        trial_classes[i] = spikes[indices[0], -1]
        
        while current_time_window_start + window_size <= t_end:
            rate_matrix[i, col_index] = (np.sum((spikes[indices, 1] >= current_time_window_start) * ((spikes[indices, 1] < (current_time_window_start + window_size)))))
            current_time_window_start = current_time_window_start + step_size
            if i == 0:
                time_vector[col_index] = current_time_window_start + window_size
            col_index += 1
            
    return rate_matrix / window_size, trial_classes, time_vector

@jit(nopython=True)
def firingRate(spike_times, t_start=-8, t_end=10, window_size=0.2, step_size=0.05, **kwargs):
    """
    Calculates the firing rate over time using a sliding window for a single trial.

    This JIT-compiled function counts spikes within a sliding window
    to compute the firing rate (in spikes/second).

    Args:
        spike_times (np.ndarray): A 1D array of spike times.
        t_start (float, optional): The start time for the analysis. Defaults to -8.
        t_end (float, optional): The end time for the analysis. Defaults to 10.
        window_size (float, optional): The duration of the sliding window (e.g., 0.2s).
                                       Defaults to 0.2.
        step_size (float, optional): The amount to slide the window forward in each
                                     step. Defaults to 0.05.

    Returns:
        tuple:
            - np.ndarray: A 2D array (shape 1xN) containing the firing rate
                          (spikes/sec) for each time bin.
            - np.ndarray: A 1D array (shape Nx1) containing the center time
                          for each corresponding bin.
    """
    num_bins = int(np.floor(((t_end - t_start) / step_size) - (window_size / step_size))) + 1
    spike_counts = np.zeros((1, num_bins), dtype=np.float32)
    time_vector = np.zeros(num_bins)
    col_index = 0
    current_time_window_start = t_start
    
    while current_time_window_start + window_size <= t_end:
        spike_counts[0, col_index] = (np.sum((spike_times >= current_time_window_start) * (spike_times < (current_time_window_start + window_size))))
        current_time_window_start = current_time_window_start + step_size
        
        time_vector[col_index] = current_time_window_start + window_size - step_size
        col_index += 1
        
    return spike_counts / window_size, time_vector

def mutual_information_over_time(neuron_rates_by_class, num_bins, binning_range, rate_start_index, rate_end_index, num_windows, num_grouped_classes, stimulus_probability, **kwargs):
    """
    Calculates the mutual information between firing rate and stimulus class.

    This function computes the MI for each time bin (window) in a specified
    range. It bins the firing rates, calculates the probability of
    response P(r) and the conditional probability P(r|s), and then
    computes MI using the formula:
    MI = sum_s sum_r [ P(s) * P(r|s) * log2( P(r|s) / P(r) ) ]

    Args:
        neuron_rates_by_class (list): List of 2D np.ndarrays. Each array
                                      [trials, time_bins] holds the firing
                                      rates for a specific class.
        num_bins (int): The number of bins to use for histogramming rates.
        binning_range (tuple): The (min, max) range for rate binning.
        rate_start_index (int): The starting time_bin index for MI calculation.
        rate_end_index (int): The ending time_bin index for MI calculation.
        num_windows (int): The total number of time bins (windows) to analyze.
                           (Should equal rate_end_index - rate_start_index).
        num_grouped_classes (int): The number of unique stimulus classes (after grouping).
        stimulus_probability (float): The probability of a single stimulus class,
                                      e.g., 1 / num_grouped_classes.

    Returns:
        list: A list of mutual information values (in bits), one for each
              time window analyzed.
    """
    
    all_rates_all_classes = np.vstack(neuron_rates_by_class)
    binned_data_all_classes = [np.histogram(all_rates_all_classes[:, i], bins=num_bins, range=(binning_range)) for i in range(rate_start_index, rate_end_index)]
    
    rates_grouped_classes = [np.vstack((neuron_rates_by_class[i], neuron_rates_by_class[i + 1])) for i in range(0, len(neuron_rates_by_class), 2)]
    binned_data_by_class_flat = [np.histogram(rates_grouped_classes[i][:, j], bins=num_bins, range=(binning_range)) for i in range(len(rates_grouped_classes)) for j in range(rate_start_index, rate_end_index)] 
      
    total_binned_probabilities = []
    class_binned_probabilities_flat = []
    binned_data_matrix = []
    information_per_window = []
                       
    data_index_all = 0
    for t in binned_data_all_classes:
        data_counts = t[data_index_all]
        total_counts = np.sum(data_counts)
        data_probs = data_counts / total_counts
        total_binned_probabilities.append(data_probs)
    
    data_index_class = 0
    for tupla in binned_data_by_class_flat:
        binned_data_counts = tupla[data_index_class]
        binned_data_sum = np.sum(binned_data_counts)
        binned_data_probs = binned_data_counts / binned_data_sum
        class_binned_probabilities_flat.append(binned_data_probs)
               
    num_arrays = len(class_binned_probabilities_flat)
    group_size = num_windows
    num_groups = num_arrays // group_size
    binned_data_by_class_only = []
    for i in range(num_groups):
        group = []
        for j in range(group_size):
            temp_index = i * group_size + j
            group.append(class_binned_probabilities_flat[temp_index])
        binned_data_by_class_only.append(group)
    
    for i in range(num_windows):
        row_arrays = [binned_data_by_class_only[j][i] for j in range(num_grouped_classes)]
        row_arrays.append(total_binned_probabilities[i])
        binned_data_matrix.append(row_arrays)

    for i in range(len(binned_data_matrix)):
        current_time_window_data = binned_data_matrix[i]
        p_r = current_time_window_data[-1]  # P(r)
        p_r_given_s = np.array(current_time_window_data[:-1])  # P(r|s)
        
        mutual_information = np.nansum(p_r_given_s * stimulus_probability * np.log2(p_r_given_s / p_r))                      
        information_per_window.append(mutual_information)
        
    return information_per_window

search_folder = "/Volumes/TOSHIBA EXT/Neuronas MPC Etiquetadas"
set_type = "OCPSLA"
labeled_neuron_directory = os.listdir(search_folder)

while(True):
    
    print("Alignment Menu: \n")
    print("1: Start of first stimulus")
    print("2: End of first stimulus")
    print("3: Start of second stimulus")
    print("4: End of second stimulus")
    
    alignment_choice = int(input("Press the selection key "))
    
    if alignment_choice in range(1, 5):
        break
    else:
        print("Incorrect selection")
    
neuron_firing_rates = []    
neuron_mutual_information = []
neuron_classes = []
neuron_information_dataframe = []
mutual_information_data_neurons = []

for folder_name in labeled_neuron_directory:
    
    if set_type in folder_name:
        
        print(folder_name)
        
        set_to_use = pjoin(search_folder, folder_name)
        neurons = os.listdir(set_to_use)
        
        neuron_index = -1
        
        neuron_indices = [-1, -3]
        selected_neurons = [neurons[i] for i in neuron_indices]
        
        for neuron_filename in tqdm(selected_neurons):
            if neuron_filename.find("neu") != -1 and neuron_filename.find(".csv") and neuron_filename.find("._") == -1:
                print(neuron_filename)
                
                behavioral_data = np.loadtxt(pjoin(search_folder, folder_name, neuron_filename), delimiter=",", usecols=range(31))
                
                correct_trials = np.nonzero(behavioral_data[:, 4] == 1)[0]
                
                behavioral_data_correct = behavioral_data[correct_trials, :]
                
                classes, valid_trials = np.unique(behavioral_data_correct[:, 1], return_counts=True)
                classes = np.int32(classes)
                
                if len(classes) == 0:
                    continue
                
                if len(valid_trials) < 5:
                    continue
                
                if len(classes) != 14:
                    continue
                
                neuron_spike_file_data = import_file(pjoin(search_folder, folder_name, neuron_filename), ",", 31)
                neuron_spikes_correct = [neuron_spike_file_data[i] for i in correct_trials]
                
                neuron_index += 1
                
                neuron_rates_by_class = []
                
                neuron_classes.append(classes)
                
                first_stimulation_period = [0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1, 1, 1.3, 1.3, 1.6, 1.6, 2, 2]
                second_stimulation_period = [2.2, 2.8, 2.2, 3, 2.4, 3.3, 2.6, 3.6, 2.8, 4, 3, 4.3, 3.3, 4.8]
                first_stimulus_period = [0.4, 0.6, 0.8, 1, 1.3, 1.6, 2]
                second_stimulus_period = [0.2, 0.8, 0.2, 1, 0.4, 1.3, 0.6, 1.6, 0.8, 2, 1, 2.3, 1.3, 2.8]
                delay_period = [1.2] * 14
                
                binary_analysis_vector = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                
                aligned_spike_times = []
                
                for class_i in classes:
                    
                    trial_indices = np.nonzero(behavioral_data_correct[:, 1] == class_i)[0]
                    
                    temporal_rate_matrix = np.empty((len(trial_indices), 317))
                    
                    aligned_spike_times_class = []
                    aligned_spike_times.append(aligned_spike_times_class)
                    
                    for index, trial_i in enumerate(trial_indices):
                        
                        if alignment_choice == 1:
                            aligned_times_trial = neuron_spikes_correct[trial_i] - behavioral_data_correct[trial_i, 24] + behavioral_data_correct[trial_i, 25]
                            
                        if alignment_choice == 2:
                            aligned_times_trial = neuron_spikes_correct[trial_i] - behavioral_data_correct[trial_i, 25] + behavioral_data_correct[trial_i, 25]
                            
                        if alignment_choice == 3:
                            aligned_times_trial = neuron_spikes_correct[trial_i] - behavioral_data_correct[trial_i, 26] + behavioral_data_correct[trial_i, 25]
                            
                        if alignment_choice == 4:
                            aligned_times_trial = neuron_spikes_correct[trial_i] - behavioral_data_correct[trial_i, 27] + behavioral_data_correct[trial_i, 25]
                            
                        aligned_spike_times_class.append(aligned_times_trial)
                        
                        temporal_rate_matrix[index, :], time_bins = firingRate(aligned_times_trial, t_start=-8, t_end=8, window_size=0.2, step_size=0.05)
                        
                    neuron_rates_by_class.append(temporal_rate_matrix)
                                        
                information_per_window = mutual_information_over_time(neuron_rates_by_class, 20, (0, 140), 152, 201, 49, 7, 1/7) 
                
                fig, ax = plt.subplots()
                ax.plot(time_bins[152:201], information_per_window)
                ax.set_ylim(-0.05, 0.25)
                plt.title(neuron_filename[0:7])
                ax.set_xlabel("time (s)")
                ax.set_ylabel("Information (bits)")
                plt.show()
                filename = f"{neuron_filename[0:7]}.svg"
                savepath = ""
                fig.savefig(os.path.join(savepath, filename))