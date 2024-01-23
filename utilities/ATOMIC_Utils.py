import numpy as np
from scipy import linalg
import seawater as sw
import datetime
from pycurrents.adcp.rdiraw import Multiread
from sklearn.metrics import pairwise_distances
from scipy import stats
import seaborn as sns 
import xarray as xr
import gsw

def yearday2datetime(yearday,year=2020):
    """Convert day of year to datetime
    
       Input: yearday, year
       yearday is an array 
       Here, year is defaulted to 2020
       
       Returns: datetime.datetime list with 
           -year
           -month
           -day
           -hour
           -minute
           -second
    """
    d = int(yearday)

    if d <= 31:                   # Here, we create conditions that specify what the values of month, day, and jd 
        month = 1                 # should be given a yearday value. These conditions could be repeated to account for 
        day = d                   # the whole year.
        jd = 0
    elif (d>31)&(d<=31+29):
        month = 2
        day = d-31
        jd = 31
    elif (d>31+29)&(d<=31+29+31):
        month = 3
        day = d-31-29
        jd = 31+29

    h = (yearday-day-jd)*24       # Calculate hour, minute, and second given yearday, day, and jd
    hour = int(h)
    m = ((h-hour)*60)
    minute = int(m)
    s = ((m-minute)*60)
    second = int(s)

    return datetime.datetime(year,month,day,hour,minute,second)

def ConvertTime(yearday,year=2020):
    """
        Converts yearday to datetime using the yearday2datetime function 
        Input: yearday, year
        yearday is an array
        year is defaulted to 2020
        
        Returns: an array of datetime.datetime values including: 
           -year
           -month
           -day
           -hour
           -minute
           -second
    """
    
    date_time = [yearday2datetime(yearday[i]) for i in range(len(yearday))] 
    
#     An alternative method for writing the above:

#     date_time = []
#     for i in range(len(yearday)):
#         date_time.append(yearday2datetime(yearday[i]))
            
    return np.array(date_time)

def ConvertTime2(yearday,year=2020):
    """
        Converts yearday to datetime using datetime.timedelta function 
        Input: yearday, year
        yearday is an array
        year is defaulted to 2020
        
        Returns: an array of datetime.datetime values including:
           -year
           -month
           -day
           -hour
           -minute
           -second
    """
    
    
    date_time0 = datetime.datetime(year,1,1)              # Set startup time (year,month,day)
   
    date_time = [date_time0+datetime.timedelta(days=yearday[i]-1) for i in range(len(yearday))]
    
#   Alternatively, one could write: 

#     date_time0 = datetime.datetime(year,1,1)
#     date_time = []
#     for i in range(len(yearday)):
        
#         date_time.append(date_time0+datetime.timedelta(days=yearday[i]-1))

    
    return date_time

def RoundTime(date_time):
    """Round time in datetime format to nearest minute 
        Input: date_time
        date_time is an array in datetime format 
        
        Return: date_time rounded to nearest minute 
    """
    # get timedelta
    td = datetime.timedelta(hours=date_time.hour, minutes=date_time.minute, 
                            seconds=date_time.second, microseconds=date_time.microsecond)
    # round timedelta to minute
    to_min = datetime.timedelta(minutes=round(td.total_seconds() / 60))
    
    # convert back to datetime
    date_time = datetime.datetime.combine(date_time, datetime.time(0)) + to_min
    
    return date_time

def RoundDateTimeArray(date_time):
    """Rounds date_time to nearest minute for an entire array 
        Input: date_time
        date_time is an array
        
        Return: date_time array rounded to nearest minute 
        
    """
    
    date_time = [RoundTime(date_time[i]) for i in range(len(date_time))]
    
    return date_time

def CropByLatLon(min_lon,max_lon,min_lat,max_lat,dataset):
    """Crops a dataset by specifying a lat and lon range
        Inputs: min_lon = minimum longitude
                max_lon = maximun longitude
                min_lat = minimum latitude
                max_lat = maximun latitude 
                dataset 
                
        Return: cropped_ds (the newly cropped dataset)
    """

    mask_lon = (dataset.longitude >= min_lon) & (dataset.longitude <= max_lon)
    mask_lat = (dataset.latitude >= min_lat) & (dataset.latitude <= max_lat)

    cropped_ds = dataset.where(mask_lon & mask_lat, drop=True)
    
    return cropped_ds

def GetTimeWindow(i,dataset):
    """Select data within +/- 15 min window centered at a particular observation (i)
        Input: i = nth observation
                dataset 
        Return: subset (new dataset with 30 min interval)
    """

# select data within +/- 15min window centered at time[i]
    dt = np.abs(dataset.time-dataset.isel(time=i).time)   # absolute time difference
    subset = dataset.where(dt<=np.timedelta64(15,'m'),drop=True)
    
    return subset 


def getImage(path):
    return OffsetImage(plt.imread(path))

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

def GetEarthCoordinates(sd_object):
    """
    From the RDIRAW pycurrents object sd_object,
    get Earth coordinates.

    Return
        - time [year days], np.array
        - lon  [degrees], np.array
        - lat [degrees], np.array
    """

    nav_txy = (sd_object.nav_start_txy + sd_object.nav_end_txy)/2.

    return nav_txy[:,0], nav_txy[:,1], nav_txy[:,2]

def GetVelocity(sd_object):
    """
    From the RDIRAW pycurrents object sd_object,
    get velocity in Earth coordinates.

    Return
        - u (eastward velocity), array[time,depth]
        - v (northward velocity), array[time,depth]
        - w (upward velocity), array[time,depth]
        - e (error velocity), array[time,depth]
    """
    return sd_object.vel[...,0], sd_object.vel[...,1], sd_object.vel[...,2], sd_object.vel[...,3]

def RawDataToXarray(sd):
    """Convert raw saildrone data to an xarray dataset
       Input: sd = saildrone number (last two digits ex: 26)
       Return: xarray dataset
       """
    
    #load raw data 
    datapath='/home/mlb15109/Research/ATOMIC/data/atomic_eurec4a_2020-ADCP_LTA/sd-10'+sd+'/LTA/*.LTA'
    m=Multiread(datapath,'wh')
    
    #get variables 
    raw_data = m.read()
    time, lon, lat = GetEarthCoordinates(raw_data)
    u,v,w,e = GetVelocity(raw_data)
    depth = raw_data.dep
#     Convert time to datetime format 
    date_time= ConvertTime2(time)
    
#     Round datetime to nearest minute
    rdate_time= RoundDateTimeArray(date_time)
    
#     the dataset 
#     REMINDER: ADD OTHER VARIABLES 
    data_set = xr.Dataset(data_vars={'u':(('time','depth'),u),
                                       'v':(('time','depth'),v),
                                        'w':(('time','depth'),w),
                                        'e':(('time','depth'),e),
                                       'latitude':(('time'),lat),
                                       'longitude':(('time'),lon)},
                            coords={'time':rdate_time,'depth':depth})
    
    data_set = data_set.expand_dims({"saildrone":['10'+sd]})
    
    return data_set 

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

def PlaneFitSubset(dataset,i,j=0,k=2):
    """Given a dataset, create 30 min interval subsets, calculate small angle approximation for each subset, 
       Do the plane fits for each subset given certain conditions
       
       Input: dataset, time observation (i), depth (j) here j is defaulted to 0,
       number of data points necessary for plane fit (k) here k is defaulted to 2
       
       Return: fit coefs for u and v, latc, lonc, and timec"""
    # 1. subsetting: dataset -> subset
    
    timec = dataset.isel(time=i).time
    dt = np.abs(dataset.time-timec)   # absolute time difference
    subset = dataset.where(dt<=np.timedelta64(15,'m'),drop=True)
    
    subset = subset.isel(depth=j)
    
    # 2. Calculate lonc,latc, dx, dy: small angle approxiamtion
    lonc = subset.longitude.mean()
    latc = subset.latitude.mean()
    theta = subset.latitude*np.pi/180
    dlon = subset.longitude-lonc
    dlat = subset.latitude-latc
    dx = 111e3*dlon*np.cos(theta)   # [m]  
    dy = 111e3*dlat                 # [m]
    
    u = subset.u.values
    v = subset.v.values
    e = subset.e.values
    ind_u = ~np.isnan(u)
    ind_v = ~np.isnan(v)
    ind_e = ~np.isnan(e)

    # 3. Plane Fits 
    # If there are no nans in the data then do the plane fit
    # Else if there are nans in the data such that the remaining data points is >= 2
    # Then remove the nans and do the plane fit
    # Else return nans for the fit coefs 
    if ~np.any(np.isnan(subset.u)):
        coefs_u = PlaneFit(dx.values,
                           dy.values,
                           u)
        coefs_v = PlaneFit(dx.values,
                           dy.values,
                           v)
        error = ErrorPlaneFit(dx.values,
                              dy.values,
                              e)
        
    elif np.all(np.sum(~np.isnan(u),axis=1)>=k):
        coefs_u = PlaneFit(dx.values[ind_u],dy.values[ind_u],u[ind_u])
        coefs_v = PlaneFit(dx.values[ind_v],dy.values[ind_v],v[ind_v])
        error = ErrorPlaneFit(dx.values[ind_e],dy.values[ind_e],e[ind_e])
    
    else:
        coefs_u = np.array([np.nan]*3)
        coefs_v = np.array([np.nan]*3)
        error = np.array([np.nan]*3)
   
        
    # 4. Return the fit coefs, lonc, latc, timec
    fit = dict(lonc=lonc, latc=latc, timec=timec.values, coefs_u=coefs_u, coefs_v=coefs_v, 
               error=error)
    return fit
    
def GetCoefsError(subset,j=0):
    """get coefs_u,coefs_v, and error for a given subset and depth. 
        depth is defaulted to 0"""
    lonc = np.array([])
    latc = np.array([])
    for i in range(len(subset.time)):
        fit = PlaneFitSubset(subset,i,j)
        lonc = np.hstack([lonc,fit['lonc']])
        latc = np.hstack([latc,fit['latc']])
        if i==0:
            coefs_u = fit['coefs_u']
            coefs_v = fit['coefs_v']
            error = fit['error']
        else:
            coefs_u = np.vstack([coefs_u,fit['coefs_u']])
            coefs_v = np.vstack([coefs_v,fit['coefs_v']])
            error = np.vstack([error,fit['error']])
    return coefs_u, coefs_v, latc, lonc, error

def to_dataset(coefs_u,coefs_v,lonc,latc,error,subset,j=0):
    """convert component of coefs_u, coefs_v, and error to xarray dataset"""
    
    var_dict = dict(mean_u=(['time'],coefs_u[:,0]),ux_gradient=(['time'],coefs_u[:,1]),
                     uy_gradient=(['time'],coefs_u[:,2]),mean_v=(['time'],coefs_v[:,0]),
                     vx_gradient=(['time'],coefs_v[:,1]), vy_gradient=(['time'],coefs_v[:,2]),
                     mean_u_error=(['time'],error[:,0]),ux_error=(['time'],error[:,1]),
                     uy_error=(['time'],error[:,2]))
    
    coord_dict = dict(longitude=(['time'],lonc),latitude=(['time'],latc),
                      time=(['time'],subset.time),depth=subset['depth'].isel(depth=j))
    data_set = xr.Dataset(data_vars=var_dict,coords=coord_dict)
    data_set = xr.concat([data_set],'depth')
    
# something like this would also work for adding in depth
# data_set.expand_dims('depth').assign_coords(depth=('depth',[subset['depth'].isel(depth=0)]))
                                                        
    return data_set

def GetCoefs_Dataset(subset,j=0):
    """get coefs_u and coefs_v for a given subset and depth and convert to xarray dataset. 
        depth is defaulted to 0"""
    lonc = np.array([])
    latc = np.array([])
    for i in range(len(subset.time)):
        fit = PlaneFitSubset(subset,i,j)
        lonc = np.hstack([lonc,fit['lonc']])
        latc = np.hstack([latc,fit['latc']])
        if i==0:
            coefs_u = fit['coefs_u']
            coefs_v = fit['coefs_v']
            error = fit['error']
        else:
            coefs_u = np.vstack([coefs_u,fit['coefs_u']])
            coefs_v = np.vstack([coefs_v,fit['coefs_v']])
            error = np.vstack([error,fit['error']])
    
    coefs = to_dataset(coefs_u,coefs_v,lonc,latc,error,j)
    
    return coefs

def get_stats(X):
    return np.round(np.nanmean(X),2), np.round(np.nanmedian(X),2), np.round(np.nanstd(X),2), np.round(stats.skew(X,nan_policy='omit'),2)

