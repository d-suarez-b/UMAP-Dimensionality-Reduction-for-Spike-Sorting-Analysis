#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:21:43 2025
@author: lucasbayones
"""
"""
SCRIPT FOR SINGLE-PATCH GEOMETRIC ANALYSIS (Manual F1 Grid Input)
This version loads the F1 grid from a manually specified path to ensure correctness.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from os.path import join
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.signal.windows import tukey
from scipy.spatial.distance import cdist
import umap
import hdbscan
# Silence Numba warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# ===================
# HELPER FUNCTIONS
# ===================
def multichannel_waveform_extractor(spike_file_path, raw_data, central_electrode, additional_electrodes, waveform_size=30):
    """
    Extracts multichannel waveforms for each spike detected on a central electrode.
    For each spike time, it concatenates the waveforms from the central and additional electrodes.
    """
    concatenated_waveforms = []
    final_spike_times = []
    
    # Channels from which to extract data, with the central one always first.
    channels_to_use = [central_electrode] + sorted(list(set(additional_electrodes)))
    
    with h5py.File(spike_file_path, "r") as f:
        central_key = "elec_" + str(central_electrode)
        if central_key not in f["spiketimes"]:
            return np.array([]), np.array([]) # No spikes on the central channel
        # Use ONLY the times from the central electrode as a reference
        ref_spike_times = np.array(f["spiketimes"][central_key])
        
        for t in ref_spike_times:
            k = int(t)
            
            # Check that the window does not go out of bounds for ANY channel
            # (We assume all channels have the same length)
            if k <= waveform_size or k >= (raw_data.shape[1] - waveform_size - 1):
                continue
            
            multichannel_snippet = []
            for channel_id in channels_to_use:
                original_id = channel_id if channel_id <= 125 else channel_id + 2  # Key adjustment
                channel_snippet = raw_data[original_id, k - waveform_size : k + waveform_size + 1]
                multichannel_snippet.append(channel_snippet)
            
            # Concatenate the snippets from each channel into a single long vector
            long_waveform = np.concatenate(multichannel_snippet)
            concatenated_waveforms.append(long_waveform)
            final_spike_times.append(t)
            
    return np.array(final_spike_times), np.array(concatenated_waveforms)

def spike_aligner(spikes, upsample_rate=8, window_length=30, alpha=0.35):
    n_sample_points = np.shape(spikes)[1]
    sample_points = np.arange(n_sample_points)
    dense_sample_points = np.arange(0, n_sample_points, 1 / upsample_rate)
    interpolator = PchipInterpolator(sample_points, spikes, axis=1)
    spikes_dense = interpolator(dense_sample_points)
    min_index = np.argmin(spikes_dense, axis=1)
    window = tukey(n_sample_points * upsample_rate, alpha=alpha)
    spikes_tukeyed = spikes_dense * window
    center = 12 * upsample_rate
    spikes_aligned_dense = np.zeros(np.shape(spikes_tukeyed))
    for count, row in enumerate(spikes_tukeyed):
        spikes_aligned_dense[count] = np.roll(row, -min_index[count] + center)
    downsample_points = np.arange(0, n_sample_points * upsample_rate, upsample_rate)
    spikes_aligned = spikes_aligned_dense[:, downsample_points]
    return spikes_aligned

def f_scorer(precision, recall):
    if precision + recall == 0: return 0
    return (2 * precision * recall) / (precision + recall)

def get_precision(sp_times, labels, gt_times, delta=30):
    unique_labs = np.unique(labels)
    f_score = []
    for k in unique_labs:
        idxs = np.where(labels == k)[0]
        spike_times = sp_times[idxs]
        if spike_times.size == 0 or gt_times.size == 0:
            f_score.append(0)
            continue
        diff_matrix = cdist(spike_times[:, None], gt_times[:, None]).astype(int)
        num_close = np.sum(diff_matrix < delta)
        precision = num_close / len(spike_times) if len(spike_times) > 0 else 0
        recall = num_close / len(gt_times) if len(gt_times) > 0 else 0
        f_score.append(f_scorer(precision * 100, recall * 100))
    return np.array(f_score)

def correct_electrode_indices(electrode_map):
    """
    Corrects the electrode indices in the map.
    For any index > 128, the real value is index - 2.
    """
    print("-> Applying index correction: electrodes > 128 will be adjusted by subtracting 2.")
    # We use NumPy's boolean indexing for an efficient operation
    corrected_map = electrode_map.copy()
    indices_to_correct = corrected_map >= 128
    corrected_map[indices_to_correct] -= 2
    return corrected_map

# =============
# CORE LOGIC
# =============
def define_geometric_rings(electrode_map, center_coords):
    y_center, x_center = center_coords
    padded_map = np.pad(electrode_map, ((2, 2), (2, 2)), mode='constant', constant_values=-1)
    y_pad, x_pad = y_center + 2, x_center + 2
    hotspot_map = padded_map[y_pad - 2 : y_pad + 3, x_pad - 2 : x_pad + 3]
    coord_rings = [
        [(2, 2)], [(2, 1), (1, 2), (2, 3), (3, 2)], [(1, 1), (1, 3), (3, 3), (3, 1)],
        [(0, 2), (2, 0), (2, 4), (4, 2)],
        [(0, 1), (1, 0), (0, 3), (3, 0), (1, 4), (4, 1), (4, 3), (3, 4)],
        [(0, 0), (0, 4), (4, 4), (4, 0)]
    ]
    electrode_rings = []
    for ring in coord_rings:
        current_ring = [hotspot_map[r, c] for r, c in ring if hotspot_map[r, c] != -1]
        electrode_rings.append(current_ring)
    return electrode_rings

def cumulative_geometric_umap_analysis(patch_info, electrode_map, threshold=None):
    patch_name, data_path, offset, ext_spikes_path, spike_times_gt, f1_matrix = patch_info
    print(f"\nProcessing {patch_name} with MULTI-CHANNEL geometric rings...") # Updated title
    
    y_center, x_center = np.unravel_index(np.nanargmax(f1_matrix), f1_matrix.shape)
    
    central_electrode_idx = electrode_map[y_center, x_center]
    print(f"** Central electrode (hotspot) identified with index: {central_electrode_idx} **")
    
    rings = define_geometric_rings(electrode_map, (y_center, x_center))
    
    raw_data = np.memmap(data_path, dtype='uint16', offset=offset, mode='r')
    num_samples = len(raw_data) // 256
    raw_data = np.array(raw_data.reshape(num_samples, 256), dtype="float32").T
    aligned_raw_data = (raw_data - (2**15 - 1)) * 0.1042
    
    # Filtering would be applied here if included
    
    accumulated_electrodes = []
    f1_scores_per_step = []
    num_electrodes_per_step = []
    
    for i, ring in enumerate(rings):
        accumulated_electrodes.extend(ring)
        unique_electrodes = sorted(list(dict.fromkeys(accumulated_electrodes)))
        num_electrodes_per_step.append(len(unique_electrodes))
        
        # --- MODIFIED LOGIC ---
        # The central electrode is the reference, the others are additional.
        ref_electrode = central_electrode_idx
        additional_electrodes = [e for e in unique_electrodes if e != ref_electrode]
        
        print(f"  Step {i+1}: Analyzing {len(unique_electrodes)} channels (Ref: {ref_electrode} + {len(additional_electrodes)} others)")
        
        # We call the new function
        times, waveforms = multichannel_waveform_extractor(ext_spikes_path, aligned_raw_data, ref_electrode, additional_electrodes)
        
        if waveforms.shape[0] < 50:
            f1_scores_per_step.append(np.nan)
            continue
            
        # Pre-processing is now applied to the concatenated "super-waveforms"
        if waveforms.ndim == 2 and waveforms.shape[1] > 5:
            filtered_waveforms = savgol_filter(waveforms, 5, 3, axis=1)
        else:
            filtered_waveforms = waveforms
            
        if i == 0:  # Only first ring
            aligned_waveforms = spike_aligner(filtered_waveforms, upsample_rate=8, window_length=30)
            waveforms_for_umap = aligned_waveforms[:, :45]  # Truncate after aligning
        else:
            waveforms_for_umap = filtered_waveforms  # Or adapt for multichannel
            
        reducer = umap.UMAP(
            min_dist=0, n_neighbors=4, n_components=2, n_epochs=2000,
            random_state=0, metric="euclidean"
        ).fit(waveforms_for_umap)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30, min_samples=5, cluster_selection_epsilon=.6
        ).fit(reducer.embedding_)
        
        fscore_array = get_precision(times, clusterer.labels_, spike_times_gt)
        max_f1 = np.amax(fscore_array) if fscore_array.size > 0 else 0
        f1_scores_per_step.append(max_f1)
        print(f"    Max F1-Score: {max_f1:.2f}")
        
    return f1_scores_per_step, num_electrodes_per_step

# =======================================
# MAIN SCRIPT EXECUTION (MANUAL F1 GRID). CAN BE AUTOMATED ACROSS MEAs
# =======================================
def main():
    # --- MANUAL CONFIGURATION FOR SINGLE PATCH ANALYSIS ---
    MANUAL_PATCH_NAME = "20170803_patch1"
    MANUAL_OFFSET = 1866
    MANUAL_F1_GRID_PATH = '/Volumes/TOSHIBA EXT/MEAs - Matrices F1/20170803_patch1.npy'
    # --- END OF CONFIGURATION ---
    
    initial_directory = "/Volumes/TOSHIBA EXT/MEAs"
    
    electrode_map = None
    try:
        from distancias import distancias
        dist = np.array([i[1] for i in list(distancias.items())])
        fscore_matrix = np.zeros((16, 16), dtype="int32")
        for idx, i in enumerate(range(0, 480, 30)):
            row = np.where(np.round(dist[:, 0]) == i)[0]
            order = np.flip(np.argsort(dist[row][:, 1]))
            fscore_matrix[:, idx] = row[order].astype(int)
        electrode_map = fscore_matrix
    except Exception as e:
        print(f"CRITICAL ERROR: Could not create electrode map. Error: {e}")
        return
        
    # <-- KEY MODIFICATION: APPLY INDEX CORRECTION HERE -->
    if electrode_map is not None:
        electrode_map = correct_electrode_indices(electrode_map)
    # -----------------------------------------------------------------
    
    # --- SINGLE PATCH ANALYSIS (NO LOOP) ---
    try:
        # Construct paths for data files
        patch_path = join(initial_directory, MANUAL_PATCH_NAME)
        patch_files = os.listdir(patch_path)
        data_path = join(patch_path, patch_files[5])
        intra_spike_file_path = join(patch_path, patch_files[6])
        circus_folder = join(patch_path, patch_files[0])
        ext_spikes_path = join(circus_folder, os.listdir(circus_folder)[3])
        
        print(f"--- Running analysis for single patch: {MANUAL_PATCH_NAME} ---")
        
        # Load the F1 matrix from the manually specified path
        f1_umap_path = MANUAL_F1_GRID_PATH
        if not os.path.exists(f1_umap_path):
            print(f"Error: Manual F1 matrix file not found at {f1_umap_path}")
            return
            
        f1_matrix = np.load(f1_umap_path)
        spike_times_gt = np.load(intra_spike_file_path)
        
        if spike_times_gt.size == 0:
            print("Error: Ground truth spike file is empty.")
            return
            
        patch_info = (MANUAL_PATCH_NAME, data_path, MANUAL_OFFSET, ext_spikes_path, spike_times_gt, f1_matrix)
        
        # Now this function will receive the already corrected electrode map
        f1_scores, num_electrodes = cumulative_geometric_umap_analysis(patch_info, electrode_map, threshold=None)
        
        euclidean_distance = [0, 30, 42.42, 60, 67.08, 84.85]
        
        # --- PLOTTING FOR SINGLE PATCH ---
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(euclidean_distance, f1_scores, marker='o', linestyle='-', color='b')
        for i, txt in enumerate(f1_scores):
            if not np.isnan(txt):
                ax.annotate(f'{txt:.1f}', (euclidean_distance[i], f1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax.set_title(f'UMAP Performance for {MANUAL_PATCH_NAME}', fontsize=16)
        ax.set_xlabel('Number of Electrodes Included (Geometric Rings)', fontsize=12)
        ax.set_ylabel('Maximum F1-Score', fontsize=12)
        if num_electrodes:
            ax.set_xticks(euclidean_distance)
        ax.grid(True)
        ax.set_ylim(bottom=0, top=105)
        
        output_filename = f'single_run_performance_{MANUAL_PATCH_NAME}.svg'
        plt.savefig(output_filename, bbox_inches='tight')
        plt.show()
        
        print(f"\nAnalysis complete. Plot saved as {output_filename}")
        
    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the analysis for {MANUAL_PATCH_NAME}.")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        print("Please check file paths and ensure the patch name is correct.")

if __name__ == '__main__':
    main()
