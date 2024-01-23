import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import datetime
import pandas as pd
import gsw
# import seaborn as sns
from Utils import *
from ATOMIC_Utils import *

import seawater as sw
from sklearn.metrics import pairwise_distances
from scipy import stats
from pycurrents.adcp.rdiraw import Multiread


def ErrorPlaneFit(x,y,sigma):
    """
        sigma: standard error in data plane fit [units of data]
    """
    sigma = sigma.flatten()
    A = np.ones((x.size,3))
    A[:,1] = x.flatten()
    A[:,2] = y.flatten()

    N = sigma[:,np.newaxis]
    SIGMA = N@N.T

    B = linalg.inv(A.T@A)

    P = B@A.T@SIGMA@A@B
    variance = np.diag(P)
    error = np.sqrt(variance)

    return error

def PlaneFit(x,y,data):
    """
        Least-squares place fit
        of data with x,y as relative
        distance to the plane center.
        
    Returns:
        - C are the coefficients of the least-squares fit
            C[:,0] is the mean, C[:,1] is the x-gradient and
            C[:,2] is the y-gradient.
    """ 
    A = np.ones((data.size,3))
    A[:,1] = x.flatten()
    A[:,2] = y.flatten()
    data = np.array(data.flatten()).T

    S = linalg.inv(A.T@A)

    C = S@A.T@data

    return C

def GetDataIntervals(dataset,i,interval):
    
    """
    Given an xarray dataset, select a time position (i), calculate change in time (dt), and subset the dataset into time intervals
    given by the interval value in minutes
    
    Inputs:
    dataset - xarray dataset
    i - time observation
    interval - value to subset dataset by (e.g. 15 min creates a 30 min time interval)
    
    Outputs:
    subset - the partitioned xarray dataset 
    """
    
    timec = dataset.isel(time=i).time
    dt = np.abs(dataset.time-timec)   # absolute time difference
    subset = dataset.where(dt<=np.timedelta64(interval,'m'),drop=True)
    
    return subset, timec

def SmallAngleApproximation(subset):
    # 2. Calculate lonc,latc, dx, dy: small angle approxiamtion
    lonc = subset.longitude.mean()
    latc = subset.latitude.mean()
    theta = subset.latitude*np.pi/180
    dlon = subset.longitude-lonc
    dlat = subset.latitude-latc
    dx = 111e3*dlon*np.cos(theta)   # [m]  
    dy = 111e3*dlat                 # [m]
    
    return lonc,latc,theta,dlon,dlat,dx,dy

def GetCoefs(subset,variable,dx,dy,k):
    
    var = subset[variable].values
    ind_var = ~np.isnan(var)
    
    if ~np.any(np.isnan(subset[variable])):
        coefs = PlaneFit(dx.values,
                        dy.values,
                        var)
        
    elif np.all(np.sum(~np.isnan(var),axis=1)>=k):
        coefs = PlaneFit(np.array(dx.values[ind_var]),np.array(dy.values[ind_var]),var[ind_var])
    
    else:
        coefs = np.array([np.nan]*3)
        
    return coefs

def GetErrorCoefs(subset,variable,dx,dy,k):
    
    
    var = subset[variable].values
    ind_var = ~np.isnan(var)
    
    if ~np.any(np.isnan(subset[variable])):

        error = ErrorPlaneFit(dx.values,
                              dy.values,
                              var)
        
    elif np.all(np.sum(~np.isnan(var),axis=1)>=k):
    
        error = ErrorPlaneFit(dx.values[ind_var],dy.values[ind_var],var[ind_var])
    
    else:
        error = np.array([np.nan]*3)
    
    return error
   
    
def GetGradient(dataset,variable,i,interval,k,error=False):
    
    subset,timec = GetDataIntervals(dataset,i=i,interval=interval)
    
    # subset = subset.isel(depth=j)
    
    lonc,latc,theta,dlon,dlat,dx,dy = SmallAngleApproximation(subset)
    
    coefs = GetCoefs(subset,variable,dx=dx,dy=dy,k=k)
    
    if error == True:
        
        coefs = GetErrorCoefs(subset,variable,dx=dx,dy=dy,k=k)
    
    fit = dict(lonc=lonc, latc=latc, timec=timec.values, coefs=coefs)
    
    return fit
    

    
def GetGradientError(dataset,variable,i,interval,k):
    
    subset,timec = GetDataIntervals(dataset,i=i,interval=interval)
    
    lonc,latc,theta,dlon,dlat,dx,dy = SmallAngleApproximation(subset)
    
    coefs = GetErrorCoefs(subset,variable,dx=dx,dy=dy,k=k)
    
    fit = dict(lonc=lonc, latc=latc, timec=timec.values, coefs=coefs)
    
    return fit
    
    
def GradientDict_to_Dataset(coefs, var1, var2, var3, lonc, latc, subset, j):
    
    # mean, x-gradient, y-gradient
    
    if j==True:
                    
        coord_dict = dict(longitude=(['time'],lonc.data),latitude=(['time'],latc.data),
                          time=(['time'],subset.time.data),depth=subset['depth'].data)
    elif j==False:
        
        coord_dict = dict(longitude=(['time'],lonc.data),latitude=(['time'],latc.data),
                      time=(['time'],subset.time.data))
                        
    data_set = xr.Dataset(coords=coord_dict)
    
    # mean, x-gradient, y-gradient
    data_set[var1] = (['time'],coefs[:,0])
    data_set[var2] = (['time'],coefs[:,1])
    data_set[var3] = (['time'],coefs[:,2])
    
    if j==True:
        data_set = xr.concat([data_set],'depth')
                    
    return data_set 


def Coefs_Dataset(subset,variable,var1,var2,var3,interval,k,j=0):
    
    lonc = np.array([])
    latc = np.array([])
    for i in range(len(subset.time)):
        fit = GetGradient(dataset=subset,variable=variable,i=i,interval=interval,k=k)
        lonc = np.hstack([lonc,fit['lonc']])
        latc = np.hstack([latc,fit['latc']])
        if i==0:
            coefs = fit['coefs']
        else:
            coefs = np.vstack([coefs,fit['coefs']])
            
    coefs_dataset = GradientDict_to_Dataset(coefs=coefs, var1=var1, var2=var2, var3=var3,lonc=lonc,latc=latc,subset=subset,j=j)
    
    return coefs_dataset

def Coefs_Dataset_Error(subset,variable,var1,var2,var3,interval,k,j=0):
    
    lonc = np.array([])
    latc = np.array([])
    for i in range(len(subset.time)):
        fit = GetGradientError(dataset=subset,variable=variable,i=i,interval=interval,k=k)
        lonc = np.hstack([lonc,fit['lonc']])
        latc = np.hstack([latc,fit['latc']])
        if i==0:
            coefs = fit['coefs']
        else:
            coefs = np.vstack([coefs,fit['coefs']])
            
    coefs_dataset = GradientDict_to_Dataset(coefs=coefs, var1=var1, var2=var2, var3=var3,lonc=lonc,latc=latc,subset=subset,j=j)
    
    return coefs_dataset

