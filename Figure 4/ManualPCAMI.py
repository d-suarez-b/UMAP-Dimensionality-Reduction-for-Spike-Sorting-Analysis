#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:01:36 2025
@author: lucasbayones
"""
"""
MODIFIED SCRIPT FOR SINGLE-ELECTRODE NEURONAL SORTING USING PCA + HDBSCAN.
- (...)
- MODIFICADO: Guarda el gráfico de clusteres, los rasters y las tasas de 
  disparo en la nueva carpeta "Sorting PCA".
- MODIFICADO: Guarda los tiempos de espiga de cada neurona en un CSV 
  fusionado con la psicometría en "Sorting PCA".
"""
import numpy as np
import pandas as pd
import os
from os.path import join
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.interpolate import PchipInterpolator
from scipy.signal.windows import tukey
# Silence Numba warnings if needed
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

# ==================
# HELPER FUNCTIONS
# ==================
def load_and_filter_data(waveforms_path, stime_path, strial_path, psychometrics_path):
    """
    Loads waveforms, spike times, trial numbers, and psychometrics.
    Filters to only spikes in trials of classes 11-15.
    Returns filtered spike times and waveforms.
    """
    # Load psychometrics to map trial to class
    psych_df = pd.read_csv(psychometrics_path)
    trial_to_class = dict(zip(psych_df['Trial'], psych_df['Class']))
 
    # Load per-electrode data
    waveforms = np.loadtxt(waveforms_path, delimiter=',') # Assuming each row is a waveform
    spike_times = np.loadtxt(stime_path, delimiter=',', dtype=float)
    spike_trials = np.loadtxt(strial_path, delimiter=',', dtype=int)
    unique_trials = np.unique(spike_trials)
    corrected_trials = np.arange(1, len(unique_trials)+1)
    # Filter to classes 11-15
    filtered_indices = []
    for idx, trial in enumerate(spike_trials):
        corrected_trial = corrected_trials[np.where(unique_trials==trial)[0][0]]
        trial_class = trial_to_class.get(corrected_trial, -1) # Default to -1 if not found
        if trial_class in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            filtered_indices.append(idx)
 
    if not filtered_indices:
        print("No spikes in classes 1-15. Skipping.")
        return np.array([]), np.array([]), np.array([]), {}
 
    filtered_waveforms = waveforms[filtered_indices]
    filtered_spike_times = spike_times[filtered_indices]
    filtered_spike_trials = spike_trials[filtered_indices]
    # Group spike times by trial for later use
    trial_to_spikes = {}
    for idx, trial in enumerate(filtered_spike_trials):
        if trial not in trial_to_spikes:
            trial_to_spikes[trial] = []
        trial_to_spikes[trial].append(filtered_spike_times[idx])
 
    return filtered_spike_times, filtered_waveforms, filtered_spike_trials, trial_to_spikes


def perform_sorting(waveforms):
    """
    Applies Savgol filter, PCA (10 components for clustering), and HDBSCAN for clustering.
    For visualization, computes a separate 3D PCA.
    Returns labels, number of neurons (clusters excluding noise), and 3D embedding for plotting.
    """
    if waveforms.shape[0] < 50:
        print("Not enough spikes for sorting. Returning 0 neurons.")
        return np.array([]), 0, None
 
    # Savgol filtering
    if waveforms.ndim == 2 and waveforms.shape[1] > 5:
        filtered_waveforms = savgol_filter(waveforms, 5, 3, axis=1)
    else:
        filtered_waveforms = waveforms
 
    # PCA for clustering (10 components)
    pca = PCA(n_components=10, random_state=0)
    embedding = pca.fit_transform(filtered_waveforms)
 
    # HDBSCAN clustering on 10D embedding
    # (Usando los parámetros que tenías en el script)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30, min_samples=5, cluster_selection_epsilon=0.6
    ).fit(embedding)
 
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    num_neurons = len([lab for lab in unique_labels if lab != -1]) # Exclude noise
 
    # Separate PCA for visualization (3 components)
    pca_viz = PCA(n_components=3, random_state=0)
    embedding_viz = pca_viz.fit_transform(filtered_waveforms)
 
    return labels, num_neurons, embedding_viz
def separate_spikes_by_neuron_and_trial(spike_times, spike_trials, labels, trial_to_class):
    """
    Separates spike times by neuron (cluster) and by trial.
    Returns a dict: neuron_id -> trial -> list of spike times.
    """
    neuron_trial_spikes = {}
    for idx, label in enumerate(labels):
        if label == -1: # Skip noise
            continue
        if label not in neuron_trial_spikes:
            neuron_trial_spikes[label] = {}
        trial = spike_trials[idx]
        if trial not in neuron_trial_spikes[label]:
            neuron_trial_spikes[label][trial] = []
        neuron_trial_spikes[label][trial].append(spike_times[idx])
 
    return neuron_trial_spikes


def transform_spike_times_to_seconds(neuron_trial_spikes, psych_df, sampling_rate=30000, align_col='o3'):
    """
    Transforms raw spike times (in samples) to seconds relative to stimulus onset (align_col, e.g., 'o1').
    Assumes trials in neuron_trial_spikes are integers starting from 1, and psych_df['Trial'] may be float.
    Updates the dict in place with times in seconds (can be negative before onset).
    """
    for neuron, trials in neuron_trial_spikes.items():
        for trial, spikes in trials.items():
            # Find onset for this trial
            onset_row = psych_df[psych_df['Trial'] == float(trial)]
            if onset_row.empty:
                print(f"Warning: No onset found for trial {trial}. Skipping transformation.")
                continue
            onset = onset_row[align_col].values[0]
            #Transform: time_sec = (raw_time / sampling_rate) - onset
            times_sec = (np.array(spikes) / sampling_rate) - onset
            neuron_trial_spikes[neuron][trial] = times_sec.tolist()  # Convert back to list if needed
    
    return neuron_trial_spikes


def save_sorted_neurons_to_csv(neuron_trial_spikes, psych_df, output_dir):
    """
    Guarda los tiempos de espiga de cada neurona en un archivo CSV 
    dentro de la carpeta "output_dir" (Sorting PCA).

    El formato es: [columnas de psychometrics.csv] | SpikeTime_1 | SpikeTime_2 | ...
    """
    try:
        print(f"\n--- Saving sorted neuron CSVs to: {output_dir} ---")

        # Iterar sobre cada neurona (clúster)
        for neuron_id, trials_data in neuron_trial_spikes.items():
            
            # Nombre del archivo (asumiendo que los IDs de neurona empiezan en 0)
            neuron_name = f"Neuron_{neuron_id + 1}.csv"
            save_path = join(output_dir, neuron_name)
            
            print(f"  Saving {neuron_name}...")
            
            # Preparar el DataFrame
            df_to_save = psych_df.copy()
            
            if not trials_data:
                max_spikes = 0
            else:
                max_spikes = max(len(s) for s in trials_data.values())
            
            # Crear las nuevas columnas de espigas
            spike_cols = []
            if max_spikes > 0:
                spike_cols = [f"SpikeTime_{i+1}" for i in range(max_spikes)]
                df_to_save[spike_cols] = np.nan
            
            # Poblar el DataFrame
            for trial_num, spikes in trials_data.items():
                idx = df_to_save[df_to_save['Trial'] == float(trial_num)].index
                
                if not idx.empty:
                    df_to_save.loc[idx, spike_cols[:len(spikes)]] = spikes
            
            # Guardar el archivo CSV
            df_to_save.to_csv(save_path, index=False, float_format='%.6f')
            
        print("--- CSV Save complete ---")
        
    except Exception as e:
        print(f"Error saving neuron CSVs: {e}")

def average_firing_rate_per_class(neuron_trial_spikes, psych_df, neuron_id, t_ini=-2, t_fin=5, win=0.2, stp=0.02, save_path=None, get_time=False):
    """
    MODIFICADO (V2):
    - FILTRO: Ahora solo incluye trials donde 'Hits' == 1 en psych_df.
    - ...
    - MODIFICADO: Añadido 'save_path' para guardar la figura.
    """
    if neuron_id not in neuron_trial_spikes:
        print(f"No data for neuron {neuron_id}.")
        return None
 
 
    # Group trials by class, PERO SÓLO SI SON HITS
    class_to_trials = {i: [] for i in range(1, 16)}
 
    for trial in neuron_trial_spikes[neuron_id]:
        # Buscar la fila correspondiente en psych_df
        trial_row = psych_df[psych_df['Trial'] == float(trial)]
        
        if trial_row.empty:
            continue # Trial no encontrado
            
        class_val = trial_row['Class'].values[0]
        hit_status = trial_row['Hits'].values[0]
        
        # Solo añadir el trial a la lista si la clase es válida Y es un Hit (== 1)
        if class_val in class_to_trials and hit_status == 1:
            class_to_trials[class_val].append(trial)
 
 
    # Compute time bins
    tam1 = int(np.floor(((t_fin - t_ini) / stp) - (win / stp))) + 1
    tiempo = np.zeros(tam1)
    t_ini2 = t_ini
    col = 0
    while t_ini2 + win <= t_fin:
        # Cálculo de tiempo
        tiempo[col] = t_ini2 + win - stp / 2 
        t_ini2 += stp
        col += 1
 
    # Compute average FR per class
    class_fr = {}
    print(len([item for sublist in list(class_to_trials.values()) for item in sublist]))
    full_fr_matrix = np.zeros(tam1)
    for class_val, trials in class_to_trials.items():
        if not trials:
            continue 
        fr_matrix = np.zeros((len(trials), tam1))
        for i, trial in enumerate(trials):
            spk2 = np.array(neuron_trial_spikes[neuron_id][trial])
            sumatoria = np.zeros(tam1)
            t_ini2 = t_ini
            col = 0
            while t_ini2 + win <= t_fin:
                count = np.sum((spk2 >= t_ini2) & (spk2 < (t_ini2 + win)))
                sumatoria[col] = count
                t_ini2 += stp
                col += 1
            fr_matrix[i] = sumatoria / win
        full_fr_matrix = np.vstack((full_fr_matrix, fr_matrix))
        # Average across trials
        class_fr[class_val] = np.mean(fr_matrix, axis=0)
    full_fr_matrix = np.delete(full_fr_matrix, 0, axis=0)
  
    df_matrix = pd.DataFrame(full_fr_matrix)
    fr_name = f"Fr_{neuron_id + 1}.csv"
    parent_dir = os.path.dirname(save_path)
    fr_path = join(parent_dir, fr_name)
    print("Saving Firing Rate csv in", fr_path)
    df_matrix.to_csv(fr_path, index=False, float_format='%.6f')
    if get_time == True:
        return class_fr, tiempo, class_to_trials
    else:
        return class_fr
# ============================
# CORE LOGIC (per electrode)
# ============================

def single_electrode_analysis(electrode_folder, psychometrics_path, min_fr_hz=2.0, t_ini_analysis=-2, t_fin_analysis=4, align_col='o3'):
    """
    Función de análisis principal modificada.
    Añade un filtro para descartar clústeres...
    MODIFICADO: Crea la carpeta "Sorting PCA", guarda el plot de
    clusteres y devuelve la ruta de la carpeta.
    """
    print(f"\nProcessing electrode folder: {electrode_folder}")

    waveforms_path = join(electrode_folder, "Waveforms.csv")
    stime_path = join(electrode_folder, "STime.csv")
    strial_path = join(electrode_folder, "STrial.csv")

    if not all(os.path.exists(p) for p in [waveforms_path, stime_path, strial_path, psychometrics_path]):
        print("Missing files. Skipping.")
        return 0, {}, {}, None

    spike_times, waveforms, spike_trials, trial_to_spikes = load_and_filter_data(
        waveforms_path, stime_path, strial_path, psychometrics_path 
    )

    if waveforms.size == 0:
        return 0, {}, {}, None

    
    psych_df = pd.read_csv(psychometrics_path)

    labels, num_neurons_raw, embedding_viz = perform_sorting(waveforms)

    if num_neurons_raw == 0:
         print("No clusters found by HDBSCAN.")
         return 0, {}, {}, None
     
    print(f"Detected {num_neurons_raw} raw clusters.")

 
    analysis_window_per_trial_s = t_fin_analysis - t_ini_analysis
    num_trials_in_classes_1_15 = len(psych_df[psych_df['Class'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])])
     
    total_analysis_time_s = 0.0
    if num_trials_in_classes_1_15 > 0:
        total_analysis_time_s = num_trials_in_classes_1_15 * analysis_window_per_trial_s
    else:
        print("Warning: No trials found in classes 1-15, cannot calculate avg FR. Skipping filter.")
        
        if total_analysis_time_s == 0 and num_neurons_raw > 0:
            print("Error: Spikes found but no trials in classes 1-15. Cannot proceed.")
            return 0, {}, {}, None

    unique_labels = np.unique(labels)
    labels_to_discard = []
     
    print(f"--- Filtering clusters by Avg Firing Rate (>= {min_fr_hz} Hz) ---")
    for label in unique_labels:
        if label == -1: continue
         
        total_spikes_in_cluster = np.sum(labels == label)
        overall_avg_fr = total_spikes_in_cluster / total_analysis_time_s
         
        if overall_avg_fr < min_fr_hz:
            labels_to_discard.append(label)
            print(f"  Discarding cluster {label}: Avg FR ({overall_avg_fr:.2f} Hz) < {min_fr_hz} Hz")
        else:
            print(f"  Keeping cluster {label}: Avg FR ({overall_avg_fr:.2f} Hz) >= {min_fr_hz} Hz")
     
    # Crear un nuevo array de etiquetas donde los clústeres descartados se marcan como -1 (ruido)
    filtered_labels = labels.copy()
    for label_to_discard in labels_to_discard:
        filtered_labels[labels == label_to_discard] = -1
         
    # Recalcular el número final de neuronas
    num_neurons = len([lab for lab in unique_labels if lab != -1 and lab not in labels_to_discard])
    print(f"Raw clusters: {num_neurons_raw}. Filtered neurons (>= {min_fr_hz} Hz): {num_neurons}.")
     
    # --- NUEVO: Crear carpeta de salida ---
    output_dir = join(electrode_folder, "Sorting PCA")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")
    np.savetxt(join(output_dir, "sorted_labels.csv"), filtered_labels, delimiter=",", fmt="%f")
    
    # Visualizar clústeres (usando las etiquetas filtradas)
    if embedding_viz is not None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Usar 'filtered_labels' para el color
        scatter = ax.scatter(embedding_viz[:, 0], embedding_viz[:, 1], embedding_viz[:, 2], c=filtered_labels, cmap='Spectral', alpha=0.7)
        fig.colorbar(scatter)
        ax.set_title(f'Clusters (Filtered, {num_neurons} Neurons)')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        
        # --- NUEVO: Guardar el gráfico de clústeres ---
        cluster_plot_path = join(output_dir, "PCA_Cluster_Visualization.png")
        print(f"  Saving cluster plot to {cluster_plot_path}")
        fig.savefig(cluster_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig) # Cerrar la figura

    # (psych_df ya está cargado)
    trial_to_class = dict(zip(psych_df['Trial'], psych_df['Class']))

    # Separar espigas (usando las etiquetas filtradas)
    neuron_trial_spikes = separate_spikes_by_neuron_and_trial(spike_times, spike_trials, filtered_labels, trial_to_class)

    
     
    # Transformar tiempos de espiga a segundos relativos a 'o3'
    neuron_trial_spikes = transform_spike_times_to_seconds(neuron_trial_spikes, psych_df, sampling_rate=30000, align_col=align_col)

    # Devolver el número de neuronas filtrado Y el psych_df Y el output_dir
    return num_neurons, neuron_trial_spikes, psych_df, output_dir
