from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.signal.windows import tukey
import numpy as np
from scipy.io import loadmat
#import spikeforest as sf
import matplotlib.pyplot as plt
#from busz_funcs import pk, pts_extraction, butter_bandpass_filter_zi, load_kachery
#from toposort.preprocessing import spike_denoiser as denoiser, spike_aligner as aligner
import umap
import hdbscan
from scipy.spatial.distance import cdist
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA 
#from quiroga import *
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import random
#import umap.plot
import pywt
from scipy import stats
import pickle as pkl        # probably Vik needs to install this library, 
import SpkSort as s
from os import path as p
def denoiser(spikes, window_length=5, polyorder=3):
    """
savgol_filter: Filter with a window length of 5 and a degree 3 polynomial.
    Use the defaults for all other parameters.


    Parameters
    ----------
    spikes : array, shape (n_spikes,n_sample_points)
        Array of all spikes you wish to denoise. Contains one spike per row.

    window_length : int
        The length of the filter window (i.e., the number of coefficients).

    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.

    Returns
    -------
    spikes_denoised : array, shape (n_spikes,n_sample_points)
        Array denoised all spikes after denoising. Contains one spike per row.


    TO DO: Implement more options for denoising filters.
    """
    spikes_denoised = savgol_filter(spikes, 5, 3)
    return spikes_denoised

def precision_fun(nmatch, nk):
    nfp=nk-nmatch
    precision=nmatch/(nmatch + nfp)
    return precision

def recall_fun(nmatch, nG):
    nmiss=nG-nmatch
    recall=nmatch/(nmiss + nmatch)
    return recall

def accuracy_fun(nmatch, nk, nG):
    nmiss=nG-nmatch
    nfp=nk-nmatch
    fscore=nmatch/(nmiss+ nmatch + nfp)
    return fscore

def fscore_fun(Gt, sorting, delta):
    nmatch=count_matching_events(Gt, sorting, delta=delta)
    nk=len(sorting)
    nG=len(Gt)
    nfp=nk-nmatch
    nmiss=nG-nmatch
    fscore=2*nmatch/(nmiss+ 2*nmatch + nfp )
    return fscore

def inclusionindex(Gt, sorting, delta):
    """
        In addition, this function returns the values proposed
        by the paper.
    """
    nmatch=count_matching_events(Gt, sorting, delta=delta)
    nk=len(sorting)
    nG=len(Gt)
    nfp=nk-nmatch
    nmiss=nG-nmatch
    #fscore=2*nmatch/(nmiss+ 2*nmatch + nfp )
    inclusion_idx=(nmatch/nk + nmatch/nG)/2
    return inclusion_idx

def overall_calc(matrix):
    """
        Assume rows ground truth, columns clustering
    """
    r, c=np.shape(matrix)
    ind=np.argmax(matrix, axis=0)
    rows=np.arange(r)  # in testing [0, 1, 2]
    cols=np.argmax(matrix, axis=1)
    indexes=np.flip(np.argsort([matrix[rows[i], cols[i]] for i in range(3)]))
    B=matrix[indexes]
    cols2=np.argmax(B, axis=1)
    values=[]
    count=0
    for i in range(len(cols2)):
        if not cols2[i] in values:
            values.append(cols2[i])
            count+=B[rows[i], cols2[i]]
        else:
            purge=np.setdiff1d(np.arange(c), values, assume_unique=False)
            if len(purge)==0:
                #print("finished with purge=", purge, "r, c: ", r, c, "")
                return count/r
            count+=np.max(B[rows[i]][purge] )
            values.append(np.argmax(purge))
    count/=r
    return count
    
def f_recording(spike_times, labelGt, label2, delta, return_conf_mat=False):
    labels=np.unique(labelGt)
    lab_clas=np.unique(label2)
    #lab_clas=lab_clas[lab_clas>=0]     # Ignoring noise.
    if len(lab_clas)==0 or len(labels)==0 and not(return_conf_mat) :
        return np.NAN
    elif len(lab_clas)==0 or len(labels)==0 and return_conf_mat:
        return np.NAN, np.zeros((len(labels), len(labels)))
    conf_matrix=np.zeros((len(labels), len(lab_clas)))  # GT X actual clas
    for lab_idx, label_i in enumerate(labels):
        times1=spike_times[labelGt==label_i]
        for label_j_idx, label_j in enumerate(lab_clas):
            times2=spike_times[label2==label_j]
            conf_matrix[lab_idx, label_j_idx]=fscore_fun(times1, times2, delta)
    f1_score_mean=overall_calc(conf_matrix)

    if return_conf_mat:
        return f1_score_mean, conf_matrix
    else:
        return f1_score_mean

def f_recording2(spike_times, labelGt, label2, delta, return_conf_mat=False):
    labels=np.unique(labelGt)
    lab_clas=np.unique(label2)
    #lab_clas=lab_clas[lab_clas>=0]     # Ignoring noise.
    if len(lab_clas)==0 or len(labels)==0 and not(return_conf_mat) :
        return np.NAN
    elif len(lab_clas)==0 or len(labels)==0 and return_conf_mat:
        return np.NAN, np.zeros((len(labels), len(labels)))
    conf_matrix=np.zeros((len(labels), len(lab_clas)))  # GT X actual clas
    conf_matrix_b=np.zeros((len(labels), len(lab_clas)))  # GT X actual clas
    for lab_idx, label_i in enumerate(labels):
        times1=spike_times[labelGt==label_i]
        for label_j_idx, label_j in enumerate(lab_clas):
            times2=spike_times[label2==label_j]
            conf_matrix[lab_idx, label_j_idx]=fscore_fun(times1, times2, delta)
            conf_matrix_b[lab_idx, label_j_idx]=inclusionindex(times1, times2, delta)
            
    f1_score_mean=overall_calc(conf_matrix)
    inc_ind_mean=overall_calc(conf_matrix_b)
    
    if return_conf_mat:
        return f1_score_mean, inc_ind_mean, conf_matrix, conf_matrix_b
    else:
        return f1_score_mean, inc_ind_mean
    


def count_matching_events(times1, times2, delta=10):
    """
        Taken from spikeforest _sortingcomparison.py
    """
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta) & (membership_sorted[0:-1] != membership_sorted[1:]))[0]
    if (len(inds) == 0):
        return 0
    inds2 = np.where(inds[:-1] + 1 != inds[1:])[0]
    return len(inds2) + 1


def aligner(
        spikes,
        upsample_rate=6,
        alignment='tukey',
        window_length=24,
        min_sample=7,
        alpha=0.35
):
    n_sample_points = np.shape(spikes)[1]
    sample_points=np.arange(n_sample_points)
    dense_sample_points = np.arange(0, n_sample_points, 1 / upsample_rate)
    
    interpolator = PchipInterpolator(sample_points, spikes, axis=1)
    spikes_dense = interpolator(dense_sample_points)

    if alignment == 'tukey':
        min_index = np.argmin(spikes_dense, axis=1)

        window = tukey(n_sample_points * upsample_rate, alpha=alpha)
        spikes_tukeyed = spikes_dense * window
        center = 12*upsample_rate  # make this optional later
        
        spikes_aligned_dense = np.zeros(np.shape(spikes_tukeyed))

        # We apply circular shift to the spikes so that they are all aligned
        # to their respective minimums at the center point
        for count, row in enumerate(spikes_tukeyed):
            spikes_aligned_dense[count] = np.roll(row,
                                                  -min_index[count] + center)
        # Note: It is very important that the downsampling is somehow
        #       Aligned to the minimum of each spike.
        downsample_points = np.arange(
            0,
            n_sample_points * upsample_rate,
            upsample_rate)

        spikes_aligned= spikes_aligned_dense[:, downsample_points]

    elif alignment == 'classic':
        # Todavia tengo que trabajar en este, pero el principal, tukey ya esta
        pass

    return spikes_aligned

def silente(spikes, labels, spike_times, cluster, percentage):  
    """
        This function returns a subset of data erasing a percentage
        of spike events for mimicking a silent situation where one neuron is less 
        active than the others.
        INPUTS
        spikes
        labels
        spike_times
        cluster
        percentage
        
    """
    perc=1-percentage
    index=np.where(labels==cluster)[0]
    erase=np.array(random.sample(list(index),int(len(index)*perc)))
    return np.delete(spikes,erase,0), np.delete(labels,erase), np.delete(spike_times,erase)
    
def silente2(spikes, labels, spike_times, cluster, percentage):  
    """
        This function returns a subset of data erasing a percentage
        of spike events for mimicking a silent situation where one neuron is less 
        active than the others.
        INPUTS
        spikes
        labels
        spike_times
        cluster
        percentage
        
    """
    mask=labels==cluster
    index=np.where(labels==cluster)[0]
    keep=np.array(random.sample(list(index),int(len(index)*percentage)))
    mask[keep]=0
    mask=~(mask)
    
    return spikes[mask], labels[mask], spike_times[mask]



def waveclus(spikes, n_features):
    n_spikes = spikes.shape[0]
    c = pywt.wavedec(spikes[0, :], 'haar', level=4)
    coeffs= np.concatenate((c[0], c[1], c[2], c[3], c[4]))
    
    CoefsWL = np.zeros((n_spikes, len(coeffs)))
    for ii in range(n_spikes):
        C = pywt.wavedec(spikes[ii,:], 'haar', level=4)
        CoefsWL[ii,:] = np.concatenate((C[0], C[1], C[2], C[3], C[4]))
    Features = CoefsWL.copy()
    DimFeatures= np.shape(Features)[1]

    Ptest= np.zeros(DimFeatures)
    KSstat= np.zeros(DimFeatures)
    
    for ii in range(DimFeatures):
        pd =stats.norm.fit(Features[:,ii])  #Estima parametros mu y sigma de los datos, 
                                            #ajustandolos a la distribucion normal
        mu = pd[0]
        sigma = pd[1]
        if sigma != 0:  #Esta parte se agrego por errores en algunas sesiones
            Data= (Features[:,ii]- mu)/ sigma  #Normalizacion Z-score
            KSstat[ii], Ptest[ii]= stats.kstest(Data, 'norm')  #KS test
            
        else:
            KSstat[ii], Ptest[ii]= np.zeros(1, dtype="float"), np.zeros(1, dtype='float')
            
    #Ordenamos (el menos normal va ultimo)
    OrderIndex = np.argsort(KSstat)
        
    spike_features = np.zeros((n_spikes, n_features))
    
    for ii in range(n_features):
        index= OrderIndex[DimFeatures - 1 - ii]  #index = OrderIndex(DimFeatures +1-ii);
        spike_features[:,ii] = Features[:, index]
    sf = spike_features.copy()
    return sf


def Funcionsota_wave(file,n_features,cycles,cluster):
    spikes, labels, spike_times = load_quiroga_spikes(file)
    score2=np.zeros(10)
    clusters=np.delete(np.unique(labels),cluster-1)
    score3=[]
    for m in range(cycles):
        print(m)
        score=[]
        for k in np.arange(0.1,1.1,0.1):
            #print(k)
            if k>=1:
                denoised = denoiser(spikes)
                spikes2 = aligner(denoised, alignment="tukey", window_length=30, upsample_rate=8)
                reducer = waveclus(spikes2, n_features)
                #umap_emb = reducer.embedding_.copy()
                clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, cluster_selection_epsilon=.6).fit(reducer)
                labs = clusterer.labels_
                #print(np.where(labs==1)[0].shape,np.where(labs==2)[0].shape,np.where(labs==3)[0].shape)
                #print(np.unique(labs))
                total_spikes=np.where(labels==cluster)[0].shape[0]
                
                tmp=[]
                indx_gt=np.where(labels==cluster)[0]
                times_gt=set(spike_times[indx_gt])
                #print("len gt times",len(times_gt))
                for label in np.unique(labs):
                    #print("label=",label,np.where(labs==label)[0].shape)
                    indx=np.where(labs==label)[0]
                    times_comparison=set(spike_times[indx])
                    tmp.append([label,len(indx),len(times_gt.intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[0])[0]]).intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[1])[0]]).intersection(times_comparison))])
                    #print('len intersection',len(times_gt.intersection(times_comparison)))
                score.append(tmp)
                break
  
            new_spikes,new_labels,new_spike_times = silente(spikes, labels, cluster, k)
            denoised = denoiser(new_spikes)
            spikes2 = aligner(denoised, alignment="tukey", window_length=30, upsample_rate=8)
            reducer = waveclus(spikes2,n_features)
            #umap_emb = reducer.embedding_.copy()
            clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, cluster_selection_epsilon=.6).fit(reducer)
            labs = clusterer.labels_
            #print(np.where(new_labels==1)[0].shape,np.where(new_labels==2)[0].shape,np.where(new_labels==3)[0].shape)
            #print(np.unique(labs))
            total_spikes=np.where(new_labels==cluster)[0].shape

            tmp=[]
            indx_gt=np.where(new_labels==cluster)[0]
            times_gt=set(new_spike_times[indx_gt])
            #print("len gt times",len(times_gt))
            for label in np.unique(labs):
                #print("label=",label,np.where(labs==label)[0].shape)
                indx=np.where(labs==label)[0]
                times_comparison=set(new_spike_times[indx])
                #print(len(set(spike_times[np.where(labels==clusters[0])[0]])))
                tmp.append([label,len(indx),len(times_gt.intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[0])[0]]).intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[1])[0]]).intersection(times_comparison))])
                #print('len intersection',len(times_gt.intersection(times_comparison)))
            score.append(tmp)
        score3.append(score)
    return score3#, score1, score2

def Funcionsota_wave_S(file,n_features,cycles,cluster):
    """
    Sergio's version of the Funcionsota_wave developed by Viktor Arseni

    Consider verifying with him that everything is working properly.

    
    """
    spikes, labels, spike_times = load_quiroga_spikes(file)
    score2=np.zeros(10)
    clusters=np.delete(np.unique(labels),cluster-1)
    score3=[]
    for m in range(cycles):
        print(m)
        score=[]
        for k in np.arange(0.1,1.1,0.1):
            #print(k)
            if k>=1:
                denoised = denoiser(spikes)
                spikes2 = aligner(denoised, alignment="tukey", window_length=30, upsample_rate=8)
                reducer = waveclus(spikes2, n_features)
                #umap_emb = reducer.embedding_.copy()
                clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, cluster_selection_epsilon=.6).fit(reducer)
                labs = clusterer.labels_
                #print(np.where(labs==1)[0].shape,np.where(labs==2)[0].shape,np.where(labs==3)[0].shape)
                #print(np.unique(labs))
                total_spikes=np.where(labels==cluster)[0].shape[0]
                
                tmp=[]
                indx_gt=np.where(labels==cluster)[0]
                times_gt=set(spike_times[indx_gt])
                #print("len gt times",len(times_gt))
                for label in np.unique(labs):
                    #print("label=",label,np.where(labs==label)[0].shape)
                    indx=np.where(labs==label)[0]
                    times_comparison=set(spike_times[indx])
                    tmp.append([label,len(indx),len(times_gt.intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[0])[0]]).intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[1])[0]]).intersection(times_comparison))])
                    #print('len intersection',len(times_gt.intersection(times_comparison)))
                score.append(tmp)
                break
  
            new_spikes,new_labels,new_spike_times = silente(spikes, labels, cluster, k)
            denoised = denoiser(new_spikes)
            spikes2 = aligner(denoised, alignment="tukey", window_length=30, upsample_rate=8)
            reducer = waveclus(spikes2,n_features)
            #umap_emb = reducer.embedding_.copy()
            clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, cluster_selection_epsilon=.6).fit(reducer)
            labs = clusterer.labels_
            #print(np.where(new_labels==1)[0].shape,np.where(new_labels==2)[0].shape,np.where(new_labels==3)[0].shape)
            #print(np.unique(labs))
            total_spikes=np.where(new_labels==cluster)[0].shape

            tmp=[]
            indx_gt=np.where(new_labels==cluster)[0]
            times_gt=set(new_spike_times[indx_gt])
            #print("len gt times",len(times_gt))
            for label in np.unique(labs):
                #print("label=",label,np.where(labs==label)[0].shape)
                indx=np.where(labs==label)[0]
                times_comparison=set(new_spike_times[indx])
                #print(len(set(spike_times[np.where(labels==clusters[0])[0]])))
                tmp.append([label,len(indx),len(times_gt.intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[0])[0]]).intersection(times_comparison)),len(set(spike_times[np.where(labels==clusters[1])[0]]).intersection(times_comparison))])
                #print('len intersection',len(times_gt.intersection(times_comparison)))
            score.append(tmp)
        score3.append(score)
    return score3#, score1, score2


def load_quiroga_spikes(path, sample_points=64, remove_noise=True):
    data = loadmat(path)
    labels = np.squeeze(data["spike_class"][0, 0])
    if remove_noise == True:
        noise_mask = ~np.squeeze(data["spike_class"][0, 1]).astype(bool)
        labels = labels[noise_mask]
    else:
        noise_mask = np.ones(labels.shape).astype(bool)
    spike_times = np.squeeze(data["spike_times"][0, 0])[noise_mask]
    spike_ts = -np.squeeze(data["data"])
    spikes = np.zeros((spike_times.shape[0], sample_points))
    for i, t in enumerate(spike_times):
        spikes[i] = spike_ts[t : t + sample_points]
    
    return spikes, labels, spike_times

def f_scorer(precision, recall):
    return (2 * precision * recall) / (precision + recall)
def get_precision(data, delta=0, lab_key="labs", score_key="score"):
    keys = [x for x in data.keys() if isinstance(x, int)]
    gt_times = gt
    for i in keys:
        labs = data[i][lab_key]
        unique_labs = np.unique(labs)
        rec_score = []
        prec_score = []
        f_score = []
        for k in unique_labs:
            spike_times = data[i]["pks"][labs == k]
            diff_matrix = cdist(spike_times[:, None], gt_times[:, None]).astype(int)
            precision = np.where(diff_matrix < delta)[0].shape[0] / len(spike_times)
            recall = np.where(diff_matrix < delta)[0].shape[0] / len(gt_times)
            rec_score.append(recall * 100)
            prec_score.append(precision * 100)
            if recall + precision > 0:
                f_score.append(f_scorer(precision * 100, recall * 100))
            else: 
                f_score.append(0)
        f_score = np.array(f_score)
        arg_score = np.argmax(f_score)
        data[i][score_key] = (f_score[arg_score], prec_score[arg_score], rec_score[arg_score], unique_labs[arg_score])


def Separacion_tired(path,score_file,cluster):
    score=np.load(score_file,allow_pickle=True)
    spikes, labels, spike_times = load_quiroga_spikes(path)
    clusters=np.delete(np.unique(labels),cluster-1)
    print(np.where(labels==1)[0].shape,np.where(labels==2)[0].shape,np.where(labels==3)[0].shape)
    prueba=np.zeros(10)
    
    
    for i in range(100):
        perc=0
        #print(i,'cycled')
        for entry in range(9):
            espigones=0
            perc+=0.1
            _,new_labels,__=silente(spikes,labels,cluster,perc)
            
            #print(entry,"emtry")
            #print(score[i][entry])
            match=sorted(score[i][entry], key=lambda x: x[2])[-1][2]
            presicion=match/sorted(score[i][entry], key=lambda x: x[2])[-1][1]
            recall=match/len(np.where(new_labels==cluster)[0])
            #f1_score=(2 * precision * recall) / (precision + recall)
            
                
                #break

                #print(tmp,'resultado final')
                #########################################################################################


            prueba[entry]+=(2 * presicion * recall) / (presicion + recall)


            #print("---------------------------------")

        match=sorted(score[i][9], key=lambda x: x[2])[-1][2]
        presicion=match/sorted(score[i][9], key=lambda x: x[2])[-1][1]
        recall=match/len(np.where(labels==cluster)[0])
        prueba[9]+=(2 * presicion * recall) / (presicion + recall)

            
    return prueba
