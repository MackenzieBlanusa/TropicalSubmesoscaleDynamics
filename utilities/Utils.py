# Utility methods for analysis of raw RDI ADCP data.
#
# History
#   - First written to look at ADCP data from Saildrone's
#       test mission (July 2019);
#
# Cesar B Rocha
# Summer, 2019

import numpy as np
from scipy import linalg
import seawater as sw

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

def GetIndices(time,tmin=117.297,tmax=117.689):
    """
    Given the time array, get the indices associated with
    tmin and tmax. Default [tmin,tmax] span the common
    back-and-forth transects in sd1033 and sd1035.

    Return
        ind, list(ind_tmin, ind_tmax)
    """
    return np.abs(time-tmin).argmin(), np.abs(time-tmax).argmin()

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

def ShipVelocityComponents(sd_object):
    """
    Calculate ship velocity components in earth coordinates
    given ship speed and two consecutive positions in the
    raw navigation data of the ADCP raw data object (sd_object).

    Return
        - u [eastward] ship velocity in m/s
        - v [northward] ship velocity in m/s
    """

    Sship = sd_object['rawnav']['AvgSpeed_mms']/1e3 # mm/s to m/s
    ang = (BAM2_to_degrees(sd_object['rawnav']['Heading_BAM2']))*np.pi/180

    return Sship*np.sin(ang),Sship*np.cos(ang)


def ShipVelocityComponentsOld(sd_object):
    """
    Calculate ship velocity components in earth coordinates
    given ship speed and two consecutive positions in the
    raw navigation data of the ADCP raw data object (sd_object).

    Return
        - u [eastward] ship velocity in m/s
        - v [northward] ship velocity in m/s
    """
    #
    Sship = sd_object['rawnav']['AvgSpeed_mms']/1e3 # mm/s to m/s
    lon1, lat1 = BAM4_to_degrees(sd_object.rawnav['Lon1_BAM4']), BAM4_to_degrees(sd_object.rawnav['Lat1_BAM4'])
    lon2, lat2 = BAM4_to_degrees(sd_object.rawnav['Lon2_BAM4']), BAM4_to_degrees(sd_object.rawnav['Lat2_BAM4'])

    ang = np.zeros_like(lon1)
    for i in range(lon1.size):
        _,ang[i] = sw.dist(lat=[lat1[i],lat2[i]], lon=[lon1[i],lon2[i]])

    ang *=np.pi/180 # degrees to radians
    #ang = (BAM2_to_degrees(sd_object['rawnav']['Heading_BAM2']))*np.pi/180

    return Sship*np.cos(ang),Sship*np.sin(ang)


def ShipVelocityComponents2(sd_object):
    """
    Calculate ship velocity components in ship coordinates

    Return
        - u [along heading] ship velocity in m/s
        - v [across heading] ship velocity in m/s
    """

    Sship = sd_object['rawnav']['AvgSpeed_mms']/1e3 # mm/s to m/s
    lon1, lat1 = BAM4_to_degrees(sd_object.rawnav['Lon1_BAM4']), BAM4_to_degrees(sd_object.rawnav['Lat1_BAM4'])
    lon2, lat2 = BAM4_to_degrees(sd_object.rawnav['Lon2_BAM4']), BAM4_to_degrees(sd_object.rawnav['Lat2_BAM4'])

    ang = np.zeros_like(lon1)
    for i in range(lon1.size):
        _,ang[i] = sw.dist(lat=[lat1[i],lat2[i]], lon=[lon1[i],lon2[i]])

    ang[ang>0] = -ang[ang>0]+360+90
    ang[ang<0] = -ang[ang<0]+90

    phi = sd_object.heading-ang # drift angle
    phi *=np.pi/180     # degrees to radians

    return Sship*np.cos(phi),Sship*np.sin(phi)

def BAM4_to_degrees(bam):
    """
    Covert raw position bam from
    BAM4 format to degrees lat or lon.

    Return
        - lat or lon in degrees
    """
    return bam * (180.0 / (2**31))

def BAM2_to_degrees(bam):
    """
    Covert raw position bam from
    BAM2 format to degrees lat or lon.

    Return
        - lat or lon in degrees
    """
    return bam * (180.0 / (2**15))

def CalculateDistance(lon,lat):
    """
    Calculate along-track distance given
        an array of lon,lat.
    Return
        - along-track distance [km]
    """
    return np.hstack([0,np.cumsum(sw.dist(lon=lon,lat=lat,units='km')[0])])


def PlaneFit(x,y,data):
    """
        Standard error in data [units of data]
    """
    A = np.ones((data.size,3))
    A[:,1] = x.flatten()
    A[:,2] = y.flatten()
    data = np.array(data.flatten()).T

    S = linalg.inv(A.T@A)

    C = S@A.T@data

    return C

def ErrorPlaneFit(x,y,sigma):
    """
        sigma: standard error in data plane fit [units of data]
    """
    A = np.ones((x.size,3))
    A[:,1] = x
    A[:,2] = y

    N = sigma.data[:,np.newaxis]
    SIGMA = N@N.T

    B = linalg.inv(A.T@A)

    P = B@A.T@SIGMA@A@B

    return P


def DistanceFromReferencePoint(LON,LAT,lon_ref,lat_ref):

    nd,nt = LON.shape
    X, Y = np.zeros_like(LON), np.zeros_like(LON)

    for i in range(nd):
        for j in range(nt):
            d, ang = sw.dist(lon=[lon_ref, LON[i,j]],lat=[lat_ref, LAT[i,j]],units='km')
            X[i,j], Y[i,j] = d*np.cos(ang*np.pi/180), d*np.sin(ang*np.pi/180)

    return X, Y

def VortDivStrain(X,Y,U,V,LAT,inds):

    x, y = X[inds].data.flatten()*1e3, Y[inds].data.flatten()*1e3 # km to m
    u, v = U[inds].data.flatten(), V[inds].data.flatten()

    um, ux, uy = PlaneFit(x=x,y=y,data=u)
    vm, vx, vy = PlaneFit(x=x,y=y,data=v)

    f = sw.f(LAT[inds].mean())

    return (vx-uy)/f, (ux+vy)/f, np.sqrt((ux-vy)**2 + (vx+uy)**2)/f, um+1j*vm
