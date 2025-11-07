import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import spikeforest as sf

from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import kstest
from scipy.stats.stats import pearsonr
from scipy.interpolate import CubicSpline as cs
from scipy.optimize import linear_sum_assignment

from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM


import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y

def pk(y, thres, min_dist=70, way='peak'):
    """Peak detection routine.

    Finds the numeric index of peaks in *y* by taking its first
    order difference. By using *thres* and *min_dist* parameters, it
    is possible to reduce the number of detected peaks.

    Parameters
    ----------
    y : ndarray
       1D time serie data array.
    thres : int
       Parameter controling the threshold level.
    min_dist : int, optional.
       Minimum distance between detections (peak with
       highest amplitude is preferred).
    way : str (optional)
        If 'peak' computes maximum
        If 'valley' computes minimum

    Returns
    -------
    out : list
        Array containing the numeric indexes of the peaks
    """

    # distance between points must be integer
    min_dist = int(min_dist)

    # flip signal
    if way=='valley': y = np.array([-i for i in y])

    # first order difference
    dy = np.diff(y)

    # propagate left and right values successively
    # to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)

    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    while len(zeros):
        # add pixels 2 by 2 to propagate left and
        # right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                    & (np.hstack([0., dy]) > 0.)
                    & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    peaks = [int(peaks[i]) for i in range(len(peaks))]

    return peaks

def pts_extraction(raw, pks, right=43, left=20, del_m_dist=3, del_tol=0.85):
    ''' Extract spikes from peaks

    Parameter
    ---------
        raw : ndarray
            Raw time serie data.
        pks : list
            Peaks position.
        right : int, optional
            Number of point to the right of the peak.
        left : int, optional
            Number of points to the left of the peak.
        del_m_dist : int
            If more than one peak is within the same time windows,
            the spike will be discarded.
        del_tol : float
            If both peaks detected within the same time windows are
            about the same height, they will be discarded. del_tol
            is the percentage above which the spike will be deleted.

    Return
    ------
        out1 : list
            List containing the spikes
        pks : list
            List containing the time for each spike (excluding deleted spks)

    '''
    # If first index in "pks" is encountere before position "left" in "raw"
    if pks[0] < left:
        # if last index in "pks" is beyond "right" position in "raw"
        if pks[-1] > len(raw_data)-right:
            # skip first and last one
            out = [raw[i-left:i+right+1] for i in pks[1:-1]]
        else:
            # skip only first one
            out = [raw[i-left:i+right+1] for i in pks[1:]]
    else:
        # keep all the peaks
        out = [raw[i-left:i+right+1] for i in pks]

    return np.array(out), np.array(pks)


def load_kachery(study_name, recording_name, uri):
    R = sf.load_spikeforest_recording(study_name=study_name, recording_name=recording_name, uri=uri)
    recording = R.get_recording_extractor()
    sorting_true = R.get_sorting_true_extractor()
    return recording, sorting_true

