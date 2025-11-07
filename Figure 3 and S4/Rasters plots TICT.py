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

@jit(nopython=True)
def firingRate(spike_times, t_start=-8, t_end=10, window_size=0.2, step_size=0.05, **kwargs):
    """
    Calculates the firing rate over time using a sliding window.

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
neuron_moving_bumps = []
neuron_classes = []
neuron_information_dataframe = []
neuron_f_test_dataframe = []
neuron_f_tests = []
control_f_test_significance = []
total_performance = []
neuron_spikes = []

for folder_name in labeled_neuron_directory:
    
    if set_type in folder_name:
        
        print(folder_name)
        
        set_to_use = pjoin(search_folder, folder_name)
        neurons = os.listdir(set_to_use)
        
        neuron_index = -1
        
        spikes_per_neuron = []
        neuron_spikes.append(spikes_per_neuron)
        
        neuron_indices = [-1, -3]  
        selected_neurons = [neurons[i] for i in neuron_indices]
        
        for neuron_filename in tqdm(selected_neurons):
            if neuron_filename.find("neu") != -1 and neuron_filename.find(".csv") and neuron_filename.find("._") == -1:
                print(neuron_filename)
                
                behavioral_data = np.loadtxt(pjoin(search_folder, folder_name, neuron_filename), delimiter=",", usecols=range(31))
                
                if len(behavioral_data) < 70:
                    continue
                
                classes, valid_trials = np.unique(behavioral_data[:, 1], return_counts=True)
                classes = np.int32(classes)
                
                if len(classes) == 0:
                    continue
                
                if len(valid_trials) < 5:
                    continue
                
                if len(classes) != 14:
                    continue
                
                neuron_spike_file_data = import_file(pjoin(search_folder, folder_name, neuron_filename), ",", 31)
                
                neuron_index += 1
                
                neuron_rates_by_class = []
                neuron_firing_rates.append(neuron_rates_by_class)
                
                neuron_classes.append(classes)
                
                first_stimulation_period = [0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1, 1, 1.3, 1.3, 1.6, 1.6, 2, 2]
                second_stimulation_period = [2.2, 2.8, 2.2, 3, 2.4, 3.3, 2.6, 3.6, 2.8, 4, 3, 4.3, 3.3, 4.8]
                first_stimulus_period = [0.4, 0.6, 0.8, 1, 1.3, 1.6, 2]
                second_stimulus_period = [0.2, 0.8, 0.2, 1, 0.4, 1.3, 0.6, 1.6, 0.8, 2, 1, 2.3, 1.3, 2.8]
                delay_period = [1.2] * 14
                
                binary_analysis_vector = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                
                aligned_spike_times = []
                spikes_per_neuron.append(aligned_spike_times)
                
                for class_i in classes:
                    
                    trial_indices = np.nonzero(behavioral_data[:, 1] == class_i)[0]
                    
                    temporal_rate_matrix = np.empty((len(trial_indices), 317))
                    
                    aligned_spike_times_class = []
                    aligned_spike_times.append(aligned_spike_times_class)
                    
                    for index, trial_i in enumerate(trial_indices):
                        
                        if alignment_choice == 1:
                            aligned_times_trial = neuron_spike_file_data[trial_i] - behavioral_data[trial_i, 24] + behavioral_data[trial_i, 25]
                            
                        if alignment_choice == 2:
                            aligned_times_trial = neuron_spike_file_data[trial_i] - behavioral_data[trial_i, 25] + behavioral_data[trial_i, 25]
                            
                        if alignment_choice == 3:
                            aligned_times_trial = neuron_spike_file_data[trial_i] - behavioral_data[trial_i, 26] + behavioral_data[trial_i, 25]
                            
                        if alignment_choice == 4:
                            aligned_times_trial = neuron_spike_file_data[trial_i] - behavioral_data[trial_i, 27] + behavioral_data[trial_i, 25]
                            
                        aligned_spike_times_class.append(aligned_times_trial)
                        
                        temporal_rate_matrix[index, :], time_bins = firingRate(aligned_times_trial, t_start=-8, t_end=8, window_size=0.2, step_size=0.05)
                        
                    neuron_rates_by_class.append(temporal_rate_matrix)
                
                def plot_raster_with_reports(aligned_spike_times, behavioral_data, neuron_name):
                    """
                    Plots a raster plot of spike times, grouped by class and colored by report.

                    Iterates through classes and trials to plot individual spike times.
                    Trials are colored black if the report (column 4) is 1, and red
                    otherwise.

                    Args:
                        aligned_spike_times (list): A nested list: [class][trial][spike_times].
                        behavioral_data (np.ndarray): The full behavioral data array, used
                                                      to get report status for coloring.
                        neuron_name (str): The name of the neuron (e.g., from the filename)
                                           to be used in the plot title.

                    Returns:
                        None. Displays a matplotlib plot.
                    """
                    trial_separation = 0.2
                    class_separation = 0.5
                    offset = 0
                
                    num_classes = len(aligned_spike_times)
                
                    reports_grouped_by_class = []
                    for class_idx in range(1, num_classes + 1):
                        class_reports = [row[4] for row in behavioral_data if row[1] == class_idx]
                        reports_grouped_by_class.append(class_reports)
                
                    for class_idx in range(num_classes - 1, -1, -1):
                        times_for_class = aligned_spike_times[class_idx]
                        reports_for_class = reports_grouped_by_class[class_idx]
                
                        min_time = -1
                        max_time = 3.5
                
                        for trial_idx, trial_spikes in enumerate(times_for_class):
                            filtered_trial = [t for t in trial_spikes if min_time <= t <= max_time]
                            if filtered_trial:
                                offset += trial_separation
                                report = reports_for_class[trial_idx]
                                color = "k" if report == 1 else "r"
                                plt.eventplot(filtered_trial, linewidths=0.2, lineoffsets=offset, linelengths=0.2, colors=color)
                        offset += class_separation
                
                    plt.title(neuron_name[0:7])
                    PATH = ""
                    plt.show()                  
                    
                plot_raster_with_reports(aligned_spike_times, behavioral_data, neuron_filename)
