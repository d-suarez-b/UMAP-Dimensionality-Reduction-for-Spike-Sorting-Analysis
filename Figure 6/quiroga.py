import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pywt
from scipy import stats

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


def quiroga_plotter(data, labels, alpha=0.1, linewidth=0.2, **kwargs):
    fig = plt.figure(figsize=(15,5))
    plt.plot(data[labels == 1].T, c='red', alpha=0.1, linewidth=0.2)
    plt.plot(data[labels == 2].T, c='blue', alpha=0.1, linewidth=0.2)
    plt.plot(data[labels == 3].T, c='green', alpha=0.1, linewidth=0.2)
    plt.show()

class data_embedder(dict):
    """
    Class to embedd data in a dict and make it accesible as attributes instead
    of having to do some indexing.
    """
    def __init__(self, filepath):
        data, labels, _ = load_quiroga_spikes(filepath)
        super().__init__(data=data, labels=labels)
        self.__dict__ = self


def waveclus(spikes, n_features=10):
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
