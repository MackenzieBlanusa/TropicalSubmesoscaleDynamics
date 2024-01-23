import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from OneShip_Utils import *  # file with necessary functions for the analysis
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy import signal
from numpy.fft import fft, ifft
from numpy import genfromtxt
from scipy.stats import binned_statistic

def hanning(data,N):
    """
    Create a hanning window filter
    
    Inputs: data - an array (1D timeseries or spatial data),
            N - length of data
    
    Output: data - data array with hanning window applied 
    """
    
    win = np.hanning(N)
    win = np.sqrt(N/(win**2).sum())*win     #why do we have to do this step? 
    data *= win
    
    return data

def calc_freq(N,dt):
    """
    Calculate the frequency of a given dataset
    
    Input: N - number of data points (length of dataset)
           dt - resolution of dataset 
           
    Output: freq - calculated frequency
            scale - scaling factor for frequency
    """
    
    # figure out if data length is even or odd
    # %2 gives you remainder (i.e. if N%2 = 0 then N is even)
    if N%2:
        even = False
    else:
        even = True
        
    # define scaling factor 
    scale = 1/(N*dt)
    
    # get frequency 
    if even:
        freq = np.arange(N/2+1) * scale
    else:
        freq = np.arange((N-1)/2 +1) * scale
        
    return freq, scale

def calc_spectrum(data,scale,N):
    """
    Calculate spectrum using FFT 
    
    Input: data - 1D timeseries or spatial data,
           scale - scaling factor (1/(N*dt))
           N - number of data points (length of data series)
           
    Output: spec - calculated spectrum 
    """
    # calculate FFT of dataset 
    fh = np.fft.rfft(data)
    
    # get spectrum by taking the real part of FFT 
    spec = 2*(fh*fh.conj()).real / scale / N**2
    
    # account for zeroth frequency 
    spec[0] = spec[0]/2
    
    if N%2:
        spec[-1] = spec[-1]/2
        
    return spec
    
    
def spec_var(scale,spec):
    """
    Calculate the variance of the spectrum 
    
    Input: scale - (1/(N*dt))
           spec - spectrum 
           
    Output: spec_var - spectrum variance
    
    """
    
    spec_var = scale*spec[1:].sum()
    
    return spec_var

def get_spectrum(dataset,dt):
    """
    Calculate the spectrum of a 1D timeseries / spatial series using FFT 
    
    Input: dataset - 1D data series (by time or distance)
    Output: freq - The frequency of the dataset,
            spec - spectrum of the dataset 
    """
    
    data = dataset.copy()
    data = signal.detrend(data)
    
    N = len(data)
    
    data = hanning(data,N)
    
    freq,scale = calc_freq(N,dt)
    
    spec = calc_spectrum(data,scale,N)
    
    return freq, spec

def interp_v(sd,dt):
    """
    Function for interpolating velocity from sd at various depths
    
    Inputs: sd - individual saildrone dataset with no nans 
            dt - spacing for the interpolation (based on average dx)
    """
    
    dx = dist_angle_calc(sd)[0]    # get distance in meters betweeen each point at surface 
    dt = dt                        # spacing for the interpolation in meters 
    
    dx_cum = np.hstack([0,np.cumsum(dx)])
    dx_even = np.arange(0,dx_cum[-1],dt)
    
    interp_vlist = []
    for d in np.arange(0,len(sd.depth),1):
        coord_dict = dict(distance=dx_even)
        dataset = xr.Dataset(coords=coord_dict)
        interp_v = np.interp(dx_even,dx_cum,sd.v[:,d])
        dataset['interp_v'] = (['distance'],interp_v)
        interp_vlist.append(dataset)
    interp_v = xr.concat(interp_vlist,'depth')
    
    return interp_v
    
def DistanceFacts(sd):
    """
    """
    
    dx = dist_angle_calc(sd)[0]
    
    print('Average distance between points in meters is ' + str(np.average(dx)))
    print('Minimum distance between points in meters is ' + str(np.min(dx)))
    print('Maximum distance between points in meters is ' + str(np.max(dx)))
    print('Standard deviation of the distance between points in meters is ' + str(np.std(dx)))
    
def FreqSpecAvg(ds,dt):
    """
    Function that takes the average spectrum by depth 
    """

    freq_list = []
    spec_list = []

    for d in np.arange(0,len(ds.depth),1):
        freq = get_spectrum(ds.isel(depth=d).interp_v,dt)[0]
        freq_list.append(freq)
        
        spec = get_spectrum(ds.isel(depth=d).interp_v,dt)[1]
        spec_list.append(spec)

    freq_avg = np.mean(freq_list,axis=0)
    spec_avg = np.mean(spec_list,axis=0)
    
    return freq_avg, spec_avg

# pabans function
def spectrum(data,dx=0.4,window=True):
    """
        TODO: Paban will write a useful
        docstring.
    """
   
   
    # make a shawllow copy of f
    f = data.copy()
   
    # detrend signal f
    f = signal.detrend(f)
   
    # constants
    N = len(f)
    nodd = N%2
    dx = 0.4     # km
    dk = 1./((N-1)*dx)  #resolution of the spectra, difference in wavenumber along x
   
    # apply window
    if window:
        win =  np.hanning(N)
        win =  np.sqrt(N/(win**2).sum())*win       #preserve variance, need to put it back in 
        f *= win    

    fh = np.fft.rfft(f)
    spec = 2*np.real(fh.conj()*fh)/dk/(N**2)
    spec[0] = spec[0]/2.

    if nodd:
        k = dk*np.arange((N-1)//2 + 1)
    else:
        k = dk*np.arange(N//2 + 1)
        spec[-1] = spec[-1]/2.
   
    return k, spec


def calc_var(spec,k):
    """ Compute total variance from spectrum """
    var = k[1]*spec[1:].sum()  # do not consider zeroth frequency
    return var

def avg_per_decade(k,E,nbins = 10):
    """ Averages the spectra with nbins per decade
        Parameters
        ===========
        - E is the spectrum
        - k is the original wavenumber array
        - nbins is the number of bins per decade
        Output
        ==========
        - ki: the wavenumber for the averaged spectrum
        - Ei: the averaged spectrum """

    dk = 1./nbins
    logk = np.log10(k)

    logki = np.arange(np.floor(logk.min()),np.ceil(logk.max())+dk,dk)
    Ei = np.zeros_like(logki)

    for i in range(logki.size):

        f = (logk>logki[i]-dk/2) & (logk<logki[i]+dk/2)

        if f.sum():
            Ei[i] = E[f].mean()
        else:
            Ei[i] = 0.

    ki = 10**logki
    fnnan = np.nonzero(Ei)
    Ei = Ei[fnnan]
    ki = ki[fnnan]

    return ki,Ei

