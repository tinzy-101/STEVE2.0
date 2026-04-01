from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr

#for dealing with files:
import os
import re
from scipy.io import readsav
import h5py
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import time

#for plotting (the rcParams updates are my personal perference to change font and increase fontsize)
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24,\
                     'xtick.labelsize' : 24,\
                     'ytick.labelsize' : 24,\
                     'axes.titlesize' : 24,\
                     'axes.labelsize' : 24,\
                     'date.autoformatter.minute': '%H:%M' })

import skymap_data_helper # all helper functions for downloading and parsing data
import cv2 # for contrast adjustment
from PIL import Image # for resolution increase
import importlib
importlib.reload(skymap_data_helper)
import math

from scipy.interpolate import griddata
from scipy.stats import pearsonr
from PIL import Image
from scipy.stats import pearsonr # correlation

import os # folder stuff 

import threading
import time 

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from tqdm.notebook import tqdm # since in jupyter

from matplotlib.path import Path # for polygon bounding box 

def planar_project_lat_lon(az_arr, el_arr, lat_camera, lon_camera, h, skymap_110_mask):
    '''
    Projected the azimuth and elevation arrays to the specified height.
    
    Parameters: 
        az_arr = 2D azimuth array for each pixel (NaNs ok, degrees, xarray)
        el_arr = 2D elevation array for each pixel (NaNs ok, degrees, xarray, no need to be filtered)
        lat_camera = latitude of camera (degrees) --> shape [480, 553]
        lon_camera = longitude of camera (degrees) --> shape [480, 553]
        h = height you want to project azimuth and elevation to to get latitude and longitude for each pixel
        skymap_110_mask = mask for where the latitude and longitude are NaNs in the ground truth lat110 and lon110 projections; to clip the edges of 

    Outputs:
        lat_aurora_arr = latitudes of the aurora projected to given height h (2D array same dimensions as original image, give projected latitude of each pixel)
        lon_aurora_arr = longitudes of the aurora project to give height h (2D array same dimensions as original image, give projected longitude of each pixel)
    '''

    # convert to radians + applying mask 
    el_arr = np.radians(np.array(el_arr))
    az_arr = np.radians(np.array(az_arr))
    
    # valid elevation mask (only want elevation in (5, 90) because likely no aurora at too low/too high elevations + camera distortions at higher elevations)
    el_mask = (el_arr > np.radians(5)) & (el_arr < np.radians(90)) #& np.isnan(el_arr)
    
    # combine with skymap mask: True = valid pixel
    valid_mask = el_mask & (~skymap_110_mask)
    
    # set invalid pixels to NaN
    el_arr[~valid_mask] = np.nan
        
    # horizontal distance between camera and aurora along camera's tangent plane
    d1_arr = h / np.tan(el_arr)

    # decompose horizontal distance into east and north components relative to camera tan plane
    dx_arr = d1_arr * np.sin(az_arr)
    dy_arr = d1_arr * np.cos(az_arr)

    # convert N/E offset components to (lat, lon) --> comes out in decimal degrees 
    lat_delta_arr = dy_arr / 111045 #degrees
    lon_delta_arr = dx_arr / (np.cos(np.radians(lat_camera + lat_delta_arr)) * 111321) 

    # add lat/long offset to camera's og lat/lon to get the lat/lon of the aurora at the chosen height!
    lat_aurora_arr = lat_camera + lat_delta_arr
    lon_aurora_arr = lon_camera + lon_delta_arr

    # apply the same mask again just in case (True = valid pixel so make it false)
    lat_aurora_arr[skymap_110_mask] = np.nan
    lon_aurora_arr[skymap_110_mask] = np.nan

    return lat_aurora_arr, lon_aurora_arr



def spherical_project_lat_lon(az_arr, el_arr, lat_camera, lon_camera, h):
    '''
    params: 
    az_arr = 2D azimuth array for each pixel (NaNs ok, degrees, xarray)
    el_arr = 2D elevation array for each pixel (NaNs ok, degrees, xarray, no need to be filtered)
    lat_camera = latitude of camera (degrees) --> shape [480, 553]
    lon_camera = longitude of camera (degrees) --> shape [480, 553]
    h = height you want to project azimuth and elevation to to get latitude and longitude for each pixel
    skymap_110_mask = mask for where the latitude and longitude are NaNs in the ground truth lat110 and lon110 projections

    returns: 
    lat_aurora_arr = latitudes of the aurora projected to given height h
    lon_aurora_arr = longitudes of the aurora project to give height h 
    '''

    #----- Preprocessing elevation and azimuth arrays + calculating necessary constants ------#
    # convert to radians + applying mask 
    el_arr = np.radians(np.array(el_arr))
    az_arr = np.radians(np.array(az_arr))
    
    # create elevation mask (True when valid) --> same elevation cutoff as pyaurorax 
    el_mask = (el_arr > np.radians(5)) & (el_arr < np.radians(90))
    
    # combine with skymap mask: True = valid pixel
    valid_mask = el_mask #& (~skymap_110_mask) # shouldnt need this skymap110 masking; elevation clipping should be sufficient --> if things still look bad without it, then something else is wrong 
    
    # set invalid pixels to NaN
    el_arr[~valid_mask] = np.nan

    # distance along ray to intersect with aurora circle
    R = 6371000 # avg radius of earth 
    t_aurora = -1*R*np.sin(el_arr) + np.sqrt(R**2*(np.sin(el_arr))**2 + 2*h*R + h**2) # this gives the world point P!

    # x component of the ray when it hits the aurora circle (head-on 2D perspective)
    x_aurora = t_aurora * np.cos(el_arr)

    # angle giving the arc distance of lat/lon offset from the camera 
    phi = np.arcsin(x_aurora / (R+h))

    # arc distance of the lat/long offset from the camera (needs to be decomposed)
    s = R * phi

    # horizontal distance between camera and aurora along tangent plane approximation to earth
    dx_arr = s * np.sin(az_arr) # dist along camera plane east (m)
    dy_arr = s * np.cos(az_arr) # dist along camera plane north (m)

    # convert N/E offset components to (lat, lon) --> comes out in decimal degrees 
    lat_delta_arr = dy_arr / 111045 #degrees
    lon_delta_arr = dx_arr / (np.cos(np.radians(lat_camera + lat_delta_arr)) * 111321) 
    #lat_delta_arr = dx_arr / 111045 #degrees
    #lon_delta_arr = dy_arr / (np.cos(np.radians(lat_camera + lat_delta_arr)) * 111321) 

    # add lat/long offset to camera's og lat/lon to get the lat/lon of the aurora at the chosen height!
    lat_aurora_arr = lat_camera + lat_delta_arr
    lon_aurora_arr = lon_camera + lon_delta_arr

    # apply the same mask again just in case --> removed for now for testing
    #mask_restricted = skymap_110_mask[1:, 1:]
    #lat_aurora_arr[skymap_110_mask] = np.nan
    #lon_aurora_arr[skymap_110_mask] = np.nan


    return lat_aurora_arr, lon_aurora_arr


def plot_lat_lon(yknf_rgb_asi_ds, fsmi_rgb_asi_ds, time_index, site_name_yknf, site_name_fsmi, yknf_lat, yknf_lon, fsmi_lat, fsmi_lon, h_target):
    ''' 
    Plots the projected latitude and longitude of the YKNF and FSMI, and overlays them. 
    Produces 3 plots total: YKNF, FSMI, overlaid, and saves each of the images. 

    Parameters:
        yknf_rgb_asi_ds: xarray of the yknf skymap <xarray>
        fsmi_rgb_asi_ds: xarray of the fsmi skymap <xarray>
        time_index: specific frame we are looking at from the asi ds <int>
        site_name_yknf: string to label the plots
        site_name_fsmi: string to label the plots
        yknf_lat: projected latitude array of one yknf frame <2d arr>
        yknf_lon: projected longitude array of one yknf frame <2d arr>
        fsmi_lat: projected latitude array of one fsmi frame <2d arr>
        fsmi_lon: projected longitude array of one fsmi frame <2d arr>
        h_target: height that yknf & fsmi frames were projected to <int>
        lat_lon_plots: folder to save the 3 images to 
        ax_skymap = the skymap ax object that gives the x and y limits (from the 110 projection skymap)

    Outputs:
        3 plots of the YKNF, FSMI, overlaid latitude and longitude
    '''    
    
    R_yknf = yknf_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_yknf = yknf_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_yknf = yknf_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    R_fsmi = fsmi_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_fsmi = fsmi_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_fsmi = fsmi_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    
    # Extract time and format it
    raw_time = yknf_rgb_asi_ds.times.values[time_index]
    time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # contrast adjustment: alpha=contrast, beta=brightness
    alpha = 5
    beta = 5
    rgb_yknf = np.stack([R_yknf, G_yknf, B_yknf], axis=-1)  # shape: (x, y, 3)
    rgb_fsmi = np.stack([R_fsmi, G_fsmi, B_fsmi], axis=-1)  # shape: (x, y, 3)
    
    rgb_yknf_adjusted = cv2.convertScaleAbs(rgb_yknf, alpha=alpha, beta=beta)
    rgb_fsmi_adjusted = cv2.convertScaleAbs(rgb_fsmi, alpha=alpha, beta=beta)

    # baseline yknf 110 plot limits, scale larger as projecting to larger altitudes
    # x_plot_min = 221.94 - h_target / 100000 * 25 
    # x_plot_max = 276.13 + h_target / 100000 * 25
    # y_plot_min = 47.37 - h_target / 100000 * 5
    # y_plot_max = 73.27 + h_target / 100000 * 5
    x_plot_min = 215
    x_plot_max = 275
    y_plot_min = 45
    y_plot_max = 80
    
    # yknf projected
    fig1, ax1 = plt.subplots(figsize=(8,8))
    scat1 = ax1.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    ax1.set_ylabel("Latitude (deg)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_title(f"{h_target/1000}km Projection: {site_name_yknf} – {time_str}", pad=30);
    plt.show()
    
    # fsmi projected
    fig2, ax2 = plt.subplots(figsize=(8,8))
    scat2 = ax2.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_title(f"{h_target/1000}km Projection: {site_name_fsmi} – {time_str}", pad=30);
    plt.show()
    
    # 110km  overlaid --> 
    plt.figure(figsize=(8,8))
    plt.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1, alpha=0.15) #0.15
    plt.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1, alpha=0.08) #0.08
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"Overlaid {h_target/1000}km Projection - {time_str}", pad=30)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    plt.show()

    return rgb_yknf_adjusted, rgb_fsmi_adjusted


def green_plot_lat_lon(yknf_rgb_asi_ds, fsmi_rgb_asi_ds, time_index, site_name_yknf, site_name_fsmi, yknf_lat, yknf_lon, fsmi_lat, fsmi_lon, h_target):
    ''' 
    Plots the projected latitude and longitude of the YKNF and FSMI, and overlays them. Specifically for the 10UTC event to focus on the picket fences
    Produces 3 plots total: YKNF, FSMI, overlaid, and saves each of the images. 

    Parameters:
        yknf_rgb_asi_ds: xarray of the yknf skymap <xarray>
        fsmi_rgb_asi_ds: xarray of the fsmi skymap <xarray>
        time_index: specific frame we are looking at from the asi ds <int>
        site_name_yknf: string to label the plots
        site_name_fsmi: string to label the plots
        yknf_lat: projected latitude array of one yknf frame <2d arr>
        yknf_lon: projected longitude array of one yknf frame <2d arr>
        fsmi_lat: projected latitude array of one fsmi frame <2d arr>
        fsmi_lon: projected longitude array of one fsmi frame <2d arr>
        h_target: height that yknf & fsmi frames were projected to <int>
        lat_lon_plots: folder to save the 3 images to 
        ax_skymap = the skymap ax object that gives the x and y limits (from the 110 projection skymap)

    Outputs:
        3 plots of the YKNF, FSMI, overlaid latitude and longitude
    '''    
    
    R_yknf = yknf_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_yknf = yknf_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_yknf = yknf_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    R_fsmi = fsmi_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_fsmi = fsmi_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_fsmi = fsmi_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    
    # Extract time and format it
    raw_time = yknf_rgb_asi_ds.times.values[time_index]
    time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # # contrast adjustment: alpha=contrast, beta=brightness
    # alpha = 5
    # beta = 5
    rgb_yknf = np.stack([R_yknf, G_yknf, B_yknf], axis=-1)  # shape: (x, y, 3)
    rgb_fsmi = np.stack([R_fsmi, G_fsmi, B_fsmi], axis=-1)  # shape: (x, y, 3)
    
    # rgb_yknf_adjusted = cv2.convertScaleAbs(rgb_yknf, alpha=alpha, beta=beta)
    # rgb_fsmi_adjusted = cv2.convertScaleAbs(rgb_fsmi, alpha=alpha, beta=beta)

    alpha_green = 2.0  # higher contrast for Green
    beta_green = 1.0  # brightness for Green
    
    # apply adjustment ONLY to index 1 (green)
    rgb_yknf[:, :, 1] = cv2.convertScaleAbs(rgb_yknf[:, :, 1], alpha=alpha_green, beta=beta_green)
    rgb_fsmi[:, :, 1] = cv2.convertScaleAbs(rgb_fsmi[:, :, 1], alpha=alpha_green, beta=beta_green)

    # decrease brightness and contast for ONLY green
    # rgb_yknf[:, :, [0, 2]] = cv2.convertScaleAbs(rgb_yknf[:, :, [0, 2]], alpha=1.5, beta=0)
    # rgb_fsmi[:, :, [0, 2]] = cv2.convertScaleAbs(rgb_fsmi[:, :, [0, 2]], alpha=1.5, beta=0)

    # baseline yknf 110 plot limits, scale larger as projecting to larger altitudes
    # x_plot_min = 221.94 - h_target / 100000 * 25 
    # x_plot_max = 276.13 + h_target / 100000 * 25
    # y_plot_min = 47.37 - h_target / 100000 * 5
    # y_plot_max = 73.27 + h_target / 100000 * 5
    x_plot_min = 215
    x_plot_max = 275
    y_plot_min = 45
    y_plot_max = 80
    
    # yknf projected
    fig1, ax1 = plt.subplots(figsize=(8,8))
    scat1 = ax1.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf.reshape(-1, 3)/256,s=1)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    ax1.set_ylabel("Latitude (deg)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_title(f"{h_target/1000}km Projection: {site_name_yknf} – {time_str}", pad=30);
    plt.show()
    
    # fsmi projected
    fig2, ax2 = plt.subplots(figsize=(8,8))
    scat2 = ax2.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi.reshape(-1, 3)/256,s=1)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_title(f"{h_target/1000}km Projection: {site_name_fsmi} – {time_str}", pad=30);
    plt.show()
    
    # 110km  overlaid --> 
    plt.figure(figsize=(8,8))
    plt.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf.reshape(-1, 3)/256,s=1, alpha=0.15) #0.15
    plt.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi.reshape(-1, 3)/256,s=1, alpha=0.08) #0.08
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"Overlaid {h_target/1000}km Projection - {time_str}", pad=30)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    plt.show()

    return rgb_yknf, rgb_fsmi


    
def mod_plot_lat_lon(yknf_rgb_asi_ds, fsmi_rgb_asi_ds, time_index, site_name_yknf, site_name_fsmi, yknf_lat, yknf_lon, fsmi_lat, fsmi_lon, h_target):
    ''' 
    Returns the contrast-adjusted rgb arrays for yknf and fsmi--> no actual plotting lol 
    
    Parameters:
        yknf_rgb_asi_ds: xarray of the yknf skymap <xarray>
        fsmi_rgb_asi_ds: xarray of the fsmi skymap <xarray>
        time_index: specific frame we are looking at from the asi ds <int>
        site_name_yknf: string to label the plots
        site_name_fsmi: string to label the plots
        yknf_lat: projected latitude array of one yknf frame <2d arr>
        yknf_lon: projected longitude array of one yknf frame <2d arr>
        fsmi_lat: projected latitude array of one fsmi frame <2d arr>
        fsmi_lon: projected longitude array of one fsmi frame <2d arr>
        h_target: height that yknf & fsmi frames were projected to <int>
        lat_lon_plots: folder to save the 3 images to 

    Outputs:
        3 plots of the YKNF, FSMI, overlaid latitude and longitude
    '''    
    
    R_yknf = yknf_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_yknf = yknf_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_yknf = yknf_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    R_fsmi = fsmi_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_fsmi = fsmi_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_fsmi = fsmi_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    
    # Extract time and format it
    raw_time = yknf_rgb_asi_ds.times.values[time_index]
    time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # contrast adjustment: alpha=contrast, beta=brightness
    alpha = 5
    beta = 5
    rgb_yknf = np.stack([R_yknf, G_yknf, B_yknf], axis=-1)  # shape: (x, y, 3)
    rgb_fsmi = np.stack([R_fsmi, G_fsmi, B_fsmi], axis=-1)  # shape: (x, y, 3)
    
    rgb_yknf_adjusted = cv2.convertScaleAbs(rgb_yknf, alpha=alpha, beta=beta)
    rgb_fsmi_adjusted = cv2.convertScaleAbs(rgb_fsmi, alpha=alpha, beta=beta)
    
    return rgb_yknf_adjusted, rgb_fsmi_adjusted



def lon_line_plot(yknf_rgb_asi_ds, fsmi_rgb_asi_ds, time_index, site_name_yknf, site_name_fsmi, yknf_lat, yknf_lon, fsmi_lat, fsmi_lon, h_target, lon_to_slice):
    ''' 
    Plots the projected latitude and longitude of the YKNF and FSMI, overlaid. 
    Plots a line down the longitude that we are slicing at (for spherical_intensity_slice_plots). 

    Parameters:
        yknf_rgb_asi_ds: xarray of the yknf skymap <xarray>
        fsmi_rgb_asi_ds: xarray of the fsmi skymap <xarray>
        time_index: specific frame we are looking at from the asi ds <int>
        site_name_yknf: string to label the plots
        site_name_fsmi: string to label the plots
        yknf_lat: projected latitude array of one yknf frame <2d arr>
        yknf_lon: projected longitude array of one yknf frame <2d arr>
        fsmi_lat: projected latitude array of one fsmi frame <2d arr>
        fsmi_lon: projected longitude array of one fsmi frame <2d arr>
        h_target: height that yknf & fsmi frames were projected to <int>
        lat_lon_plots: folder to save the 3 images to 
        ax_skymap = the skymap ax object that gives the x and y limits (from the 110 projection skymap)
        lon_to_slice = the longitude we sliced at 

    Outputs:
        3 plots of the YKNF, FSMI, overlaid latitude and longitude
    '''    
    
    R_yknf = yknf_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_yknf = yknf_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_yknf = yknf_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    R_fsmi = fsmi_rgb_asi_ds.image.sel(channel="R").isel(times=time_index).values
    G_fsmi = fsmi_rgb_asi_ds.image.sel(channel="G").isel(times=time_index).values
    B_fsmi = fsmi_rgb_asi_ds.image.sel(channel="B").isel(times=time_index).values
    
    
    # Extract time and format it
    raw_time = yknf_rgb_asi_ds.times.values[time_index]
    time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # contrast adjustment: alpha=contrast, beta=brightness
    alpha = 5
    beta = 5
    rgb_yknf = np.stack([R_yknf, G_yknf, B_yknf], axis=-1)  # shape: (x, y, 3)
    rgb_fsmi = np.stack([R_fsmi, G_fsmi, B_fsmi], axis=-1)  # shape: (x, y, 3)
    
    rgb_yknf_adjusted = cv2.convertScaleAbs(rgb_yknf, alpha=alpha, beta=beta)
    rgb_fsmi_adjusted = cv2.convertScaleAbs(rgb_fsmi, alpha=alpha, beta=beta)

    x_plot_min = 215
    x_plot_max = 275
    y_plot_min = 45
    y_plot_max = 80

    # # overlaid 
    # plt.figure(figsize=(8,8))
    # plt.scatter(yknf_lon.flatten(),yknf_lat.flatten(),c=rgb_yknf_adjusted.reshape(-1, 3)/256,s=1, alpha=0.15) #0.15
    # plt.scatter(fsmi_lon.flatten(),fsmi_lat.flatten(),c=rgb_fsmi_adjusted.reshape(-1, 3)/256,s=1, alpha=0.08) #0.08
    # plt.xlabel("Longitude (deg)")
    # plt.ylabel("Latitude (deg)")
    # plt.title(f"Overlaid {h_target/1000}km Projection - {time_str}", pad=30)
    # plt.xlim((x_plot_min, x_plot_max))
    # plt.ylim((y_plot_min, y_plot_max))
    # plt.axvline(x=lon_to_slice, color='red', linestyle='--', linewidth=2, label=f"Slice at {lon_to_slice:.2f}°")
    # plt.legend()
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        yknf_lon.flatten(),
        yknf_lat.flatten(),
        c=rgb_yknf_adjusted.reshape(-1, 3) / 256,
        s=1,
        alpha=0.15
    )
    
    ax.scatter(
        fsmi_lon.flatten(),
        fsmi_lat.flatten(),
        c=rgb_fsmi_adjusted.reshape(-1, 3) / 256,
        s=1,
        alpha=0.08
    )
    
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Overlaid {h_target/1000}km Projection - {time_str}", pad=30)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_min, y_plot_max)
    
    line = ax.axvline(
        lon_to_slice,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Slice at {lon_to_slice:.2f}°"
    )
    
    ax.legend()
    plt.show()




def compute_metrics_for_altitude(
    h_target,
    t_arr,
    full_azimuth_yknf, full_elevation_yknf, lat_yknf, lon_yknf,
    full_azimuth_fsmi, full_elevation_fsmi, lat_fsmi, lon_fsmi,
    yknf_110_mask, fsmi_110_mask,
    yknf_rgb_asi_ds, fsmi_rgb_asi_ds,
    global_lon_arr, # so same number of longitude slices for all the different projected altitudes
):
    # ---- project ONCE per altitude ----
    yknf_lat_proj, yknf_lon_proj = spherical_project_lat_lon(
        full_azimuth_yknf, full_elevation_yknf,
        lat_yknf, lon_yknf, h_target, yknf_110_mask
    )

    fsmi_lat_proj, fsmi_lon_proj = spherical_project_lat_lon(
        full_azimuth_fsmi, full_elevation_fsmi,
        lat_fsmi, lon_fsmi, h_target, fsmi_110_mask
    )

    #---------Define a single global longitude grid -------------+
    global_lon_min = int(np.floor(
        min(np.nanmin(yknf_lon_proj), np.nanmin(fsmi_lon_proj))
    ))
    global_lon_max = int(np.ceil(
        max(np.nanmax(yknf_lon_proj), np.nanmax(fsmi_lon_proj))
    ))


    # ---- overlap region (once) ----
    lat_min = min(np.nanmin(yknf_lat_proj), np.nanmin(fsmi_lat_proj))
    lon_min = min(np.nanmin(yknf_lon_proj), np.nanmin(fsmi_lon_proj))
    lat_max = max(np.nanmax(yknf_lat_proj), np.nanmax(fsmi_lat_proj))
    lon_max = max(np.nanmax(yknf_lon_proj), np.nanmax(fsmi_lon_proj))

    # fixed auroral box this is based on where the aurora falls for yknf and fsmi 
    global_lat_min_box, global_lat_max_box = 61, 65
    global_lon_min_box, global_lon_max_box = 244, 257

    ssd_vals = [] # sum of squared differences
    sad_vals = [] # sum of absolute differences
    avg_diffs = [] # average of absolute differences
    med_diffs = [] # median of absolute differences 
    diffs = []

    yknf_peaks_all = []  # store peak arrays for each time_index
    fsmi_peaks_all = []
    
    for time_index in t_arr:
        # ---- extract RGB frames ----
        yknf_rgb, fsmi_rgb = mod_plot_lat_lon(
            yknf_rgb_asi_ds, fsmi_rgb_asi_ds,
            time_index,
            "Yellowknife", "Fort Smith",
            yknf_lat_proj, yknf_lon_proj,
            fsmi_lat_proj, fsmi_lon_proj,
            h_target
        )

        # ---- interpolate (only need peak arrays) ----
        _, _, _, yknf_peak = mod_line_interpolate(
            yknf_lat_proj, yknf_lon_proj, yknf_rgb,
            lat_min, lat_max, lon_min, lon_max,
            global_lat_min_box, global_lat_max_box,
            global_lon_min_box, global_lon_max_box,
            "yknf", None, time_index, h_target, global_lon_arr
        )

        _, _, _, fsmi_peak = mod_line_interpolate(
            fsmi_lat_proj, fsmi_lon_proj, fsmi_rgb,
            lat_min, lat_max, lon_min, lon_max,
            global_lat_min_box, global_lat_max_box,
            global_lon_min_box, global_lon_max_box,
            "fsmi", None, time_index, h_target, global_lon_arr
        )

        yknf_peaks_all.append(yknf_peak)
        fsmi_peaks_all.append(fsmi_peak)

    # convert to arrays 
    yknf_peaks_all = np.array(yknf_peaks_all)
    fsmi_peaks_all = np.array(fsmi_peaks_all)

    global_valid_mask = np.isfinite(yknf_peaks_all) & np.isfinite(fsmi_peaks_all)

    for t_idx in range(len(t_arr)):
        diff = yknf_peaks_all[t_idx][global_valid_mask[t_idx]] - fsmi_peaks_all[t_idx][global_valid_mask[t_idx]]
       
        # ---- Metrics ----
        ssd_vals.append(np.nansum(np.abs(diff**2)))
        sad_vals.append(np.nansum(np.abs(diff)))
        avg_diffs.append(np.mean(np.abs(diff)))
        med_diffs.append(np.median(np.abs(diff)))
        diffs.append(np.abs(diff)) # 2d array of the differences for investigation 

    return h_target, ssd_vals, sad_vals, avg_diffs, med_diffs, diffs


# do on r, g, b channels independently
# interpolate to 1 line varying lat, single lon

def old_line_interpolate(lat_proj, lon_proj, rgb, 
                     lat_min, lat_max, lon_min, lon_max,
                     lat_min_box, lat_max_box, 
                     lon_min_box, lon_max_box,
                     site_name, folder_name, time_index, 
                     altitude):

    """
    input:
        lat_proj = 2D projected latitude arr (output of project_lat_lon())
        lon_proj = 2D projected longitude arr (output of project_lat_lon())
        rgb = 3D rgb array (output of plot_lat_lon()) 
        lat_min = minimum latitude to create grid for (should be common between the 2 cameras)
        lat_max = maximum latitude to create grid for (should be common between the 2 cameras)
        lon_min = minimum longitude to create grid for (should be common between the 2 cameras)
        lon_max = maximum longitude to create grid for (should be common between the 2 cameras)
        lat_min_box = restricted latitude for where aurora is in the frame (needed for finding max intensity) --> change for diff altitude projs if necessary
        lat_max_box = restricted latitude for where aurora is in the frame (needed for finding max intensity) --> change for diff altitude projs if necessary
        lon_min_box = restrict longitude for where aurora is in the frame (else will get meaningless peak intensities and screw up line of best fit)
        lon_max_box = restrict longitude for where aurora is in the frame (else will get meaningless peak intensities and screw up line of best fit)
        time_index = for naming purposes 

    output:
        R_peak_lat_arr = 1D arr of all the latitudes that the R-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        G_peak_lat_arr = 1D arr of all the latitudes that the G-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        B_peak_lat_arr = 1D arr of all the latitudes that the B-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        lon_arr = 1D arr of all the longitudes corresponding in idx to the peak latitude arrays (should be same length as the other 3 returned arrays)
    """

    # latitude and longitude arrays to define the interpolation grid
    lat_arr = np.arange(lat_min, lat_max, 0.5) # this is about close to the actual pixel resolution
    lon_arr = np.arange(lon_min, lon_max, 1) # 1 slice in longitude every 1 degree, maybe need to adjust this as projecting to higher altitudes

    print(f"Total {len(lon_arr)} longitudes to interpolate\n")
    print(f"Interpolating at these longitudes: {lon_arr}\n")
    print("TESTING")
    R_peak_lat_arr = []
    
    # longitudes that have a peak latitude that we are saving (to handle the masking we do on the latitude ranges)
    R_lon_arr = []

    # separate into R, G, B channels
    R = rgb[:,:, 0]
    G = rgb[:,:, 1]
    B = rgb[:,:, 2]

    # ((site_name, altitude, time_index, longitude) --> (lon_slice, lat_slice, rgb_slice)) dict, keys = longitude
    # already include h,t and site_name in multithreading: (longitude, lon_slice, lat_slice, rgb_slice)
    interp_slice_arr = []

    # ((site_name, altitude, time_index, longitude) --> (lat_arr, R_intensity_profile, G_intensity_profile, B_intensity_profile))
    # (longitude, lon_slice, lat_slice, rgb_slice)
    rgb_profile_arr = []

    # ((site_name, altitude, time_index) --> (r_lon_arr, R_peak_lat_arr)), keys=site_name, altitude_time_index
    # don't need this, just return directly 
    #peak_R_lat_lon_dict = {}
    
        
    for idx, single_lon in enumerate(lon_arr):

        # restrict to a longitude slice
        pixel_width = 1
        lon_pixel_spacing = 1
        lon_diff = np.abs(lon_proj - single_lon)
        slice_mask = lon_diff <= pixel_width * lon_pixel_spacing

        if np.sum(slice_mask) == 0:
            #print(f"No points found in slice for longitude {single_lon}, skipping...")
            continue
    
        lon_slice = lon_proj[slice_mask]
        lat_slice = lat_proj[slice_mask]
        rgb_slice = rgb[slice_mask] # needed for plotting later, not for interpolation!
        R_slice = R[slice_mask]
        G_slice = G[slice_mask]
        B_slice = B[slice_mask]

        # save for plotting outside of multithreading
        #interp_slice_dict[(site_name, altitude, time_index, single_lon)] = [lon_slice, lat_slice, rgb_slice]
        interp_slice_arr.append([single_lon, lon_slice, lat_slice, rgb_slice, lon_proj, lat_proj, rgb]) # added og lon_proj, lat_proj, rgb so plotting bounds are nicer


        # flatten yknf and fsmi projected arrays + R + G + B arrays 
        #flattened_points = np.column_stack((lon_proj.flatten(), lat_proj.flatten())) # (N,2) pairs of (lat, lon)
        flattened_points = np.column_stack((lon_slice.flatten(), lat_slice.flatten())) # (N,2) pairs of (lat, lon)
        flattened_R = R_slice.flatten()
        flattened_G = G_slice.flatten()
        flattened_B = B_slice.flatten()

        # remove NaNs
        # masks for NaNs in R, G, and B channels separately (basically remove the NaNs and infinite values if there is) --> where the projected lat, lon, or rgb values corresponding to pixel are NaNs
        nan_mask_R = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(flattened_R))
        nan_mask_G = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(flattened_G))
        nan_mask_B = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(flattened_B))

        points_R_clean = flattened_points[nan_mask_R]
        values_R_clean = flattened_R[nan_mask_R]
        
        points_G_clean = flattened_points[nan_mask_G]
        values_G_clean = flattened_G[nan_mask_G]

        points_B_clean = flattened_points[nan_mask_B]
        values_B_clean = flattened_B[nan_mask_B]

        
        #print(f"{idx}-- Longitude Interpolating: {single_lon}")
        single_lon_arr = single_lon * np.ones(np.shape(lat_arr)) # constant latitude, as many as there are latitudes to interpolate over 
        interp_locations = np.column_stack((single_lon_arr, lat_arr)) # array of (lon, lat) x-y pairs, longitude is constant so get line in lat

        # for Qhull errors (skip the )
        if len(points_R_clean) >= 3 and len(values_R_clean >= 3):   # minimum for 2D linear interpolation
            R_intensity_profile = griddata(points_R_clean, values_R_clean, interp_locations, method='linear')
        else:
            R_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        if len(points_G_clean) >= 3 and len(values_G_clean >= 3):   # minimum for 2D linear interpolation
            G_intensity_profile = griddata(points_G_clean, values_G_clean, interp_locations, method='linear')
        else:
            G_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        if len(points_B_clean) >= 3 and len(values_B_clean >= 3):   # minimum for 2D linear interpolation
            B_intensity_profile = griddata(points_B_clean, values_B_clean, interp_locations, method='linear')
        else:
            B_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        # save for plotting outside of multithreading
        #rgb_profile_dict[(site_name, altitude, time_index, single_lon)] = [lat_arr, R_intensity_profile, G_intensity_profile, B_intensity_profile]
        rgb_profile_arr.append([single_lon, lat_arr, R_intensity_profile, G_intensity_profile, B_intensity_profile])
        
        # restrict the latitude to likely latitudes for the aurora (to minimize the outliers)
        # NOT robust for different time stamps 
        # also remove NaNs so curve_fit will work (NaNs added if points to be interpolated outside of convex hull)
        lat_mask = (lat_arr >= lat_min_box - 2) & (lat_arr <= lat_max_box + 2)
        R_nan_mask = np.isfinite(R_intensity_profile)
        R_intensity_restricted = R_intensity_profile[lat_mask & R_nan_mask]
        lat_arr_restricted = lat_arr[lat_mask] 

        if R_intensity_restricted.size==0:
            #print("No points in this latitude range for this longitude slice.\n")
            continue
        
        # jsut pick out the maximum intensity from the expected range of latitudes
        # indices for intensity & latitude should correspond 
        # make another array for the corresponding longitudes to the peak latitude (accounts for when hitting the exception intensity_restricted.size=0)
        # restrict the longitude so don't get meaningless peak intensities 
        if single_lon <= lon_max_box and single_lon >= lon_min_box:
            idx_peak_R = np.argmax(R_intensity_restricted)
            R_peak_lat_arr.append(lat_arr_restricted[idx_peak_R])
            R_lon_arr.append(single_lon)
            
    # remove outliers in peak latitude before line fitting
    # need to convert to np arrays
    R_peak_lat_arr = np.asarray(R_peak_lat_arr)
    R_lon_arr = np.asarray(R_lon_arr)

    # save for plotting outside of multithreading
    #peak_R_lat_lon_dict[(site_name, altitude, time_index)] = [R_lon_arr, R_peak_lat_arr] 

    # plot aurora image
    # plt.figure(figsize=(10,8))
    # plt.scatter(lon_proj.flatten(),lat_proj.flatten(),c=rgb.reshape(-1, 3)/256,s=1)
    # plt.ylabel("Latitude (deg)")
    # plt.xlabel("Longitude (deg)")
    # plt.title(f"Aurora with Fitted Lines")
    
    # # plot the peak points for R, G, B (raw)
    # plt.scatter(R_lon_arr, R_peak_lat_arr, color='red', label='Peak R', s=30)

    # plt.legend(
    #     fontsize=8,        
    #     markerscale=0.8,   
    #     handlelength=1,    # shorter legend lines
    #     handletextpad=0.4, # tighter text spacing
    #     borderpad=0.3      # smaller legend box padding
    # )
    # plt.show()

    return (interp_slice_arr, rgb_profile_arr,
           R_lon_arr, R_peak_lat_arr)



''' fixing to get consistent longitude array size --> pick same number of longitudes to interpolate over '''
def mod_line_interpolate(lat_proj, lon_proj, rgb, 
                     lat_min, lat_max, lon_min, lon_max,
                     lat_min_box, lat_max_box, 
                     lon_min_box, lon_max_box,
                     site_name, folder_name, time_index, 
                     altitude,
                     global_lon_arr):

    """
    input:
        lat_proj = 2D projected latitude arr (output of project_lat_lon())
        lon_proj = 2D projected longitude arr (output of project_lat_lon())
        rgb = 3D rgb array (output of plot_lat_lon()) 
        lat_min = minimum latitude to create grid for (should be common between the 2 cameras)
        lat_max = maximum latitude to create grid for (should be common between the 2 cameras)
        lon_min = minimum longitude to create grid for (should be common between the 2 cameras)
        lon_max = maximum longitude to create grid for (should be common between the 2 cameras)
        lat_min_box = restricted latitude for where aurora is in the frame (needed for finding max intensity) --> change for diff altitude projs if necessary
        lat_max_box = restricted latitude for where aurora is in the frame (needed for finding max intensity) --> change for diff altitude projs if necessary
        lon_min_box = restrict longitude for where aurora is in the frame (else will get meaningless peak intensities and screw up line of best fit)
        lon_max_box = restrict longitude for where aurora is in the frame (else will get meaningless peak intensities and screw up line of best fit)
        time_index = for naming purposes 
        other_lon_proj: longitude projection from the other camera (to compute global lon bounds)

    output:
        R_peak_lat_arr = 1D arr of all the latitudes that the R-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        G_peak_lat_arr = 1D arr of all the latitudes that the G-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        B_peak_lat_arr = 1D arr of all the latitudes that the B-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        lon_arr = 1D arr of all the longitudes corresponding in idx to the peak latitude arrays (should be same length as the other 3 returned arrays)
    """
    
    #global_lon_arr = np.arange(global_lon_min, global_lon_max + 1, 1)

    # latitude and longitude arrays to define the interpolation grid
    lat_arr = np.arange(lat_min, lat_max, 0.5) # this is about close to the actual pixel resolution
    #lon_arr = np.arange(lon_min, lon_max, 1) # 1 slice in longitude every 1 degree, maybe need to adjust this as projecting to higher altitudes

    print(f"Total {len(global_lon_arr)} longitudes to interpolate\n")
    print(f"Interpolating at these longitudes: {global_lon_arr}\n")

    #R_peak_lat_arr = []
    
    # longitudes that have a peak latitude that we are saving (to handle the masking we do on the latitude ranges)
    R_lon_arr = []

    # separate into R, G, B channels
    R = rgb[:,:, 0]
    G = rgb[:,:, 1]
    B = rgb[:,:, 2]

    # ((site_name, altitude, time_index, longitude) --> (lon_slice, lat_slice, rgb_slice)) dict, keys = longitude
    # already include h,t and site_name in multithreading: (longitude, lon_slice, lat_slice, rgb_slice)
    interp_slice_arr = []

    # ((site_name, altitude, time_index, longitude) --> (lat_arr, R_intensity_profile, G_intensity_profile, B_intensity_profile))
    # (longitude, lon_slice, lat_slice, rgb_slice)
    rgb_profile_arr = []

    # ((site_name, altitude, time_index) --> (r_lon_arr, R_peak_lat_arr)), keys=site_name, altitude_time_index
    # don't need this, just return directly 
    #peak_R_lat_lon_dict = {}

    # preallocate the sizes of the R_peak_lat_array and R_lon_arr (don't <continue> when run into issue with longitude, rather just add nan in that place)
    R_peak_lat_arr = np.full(len(global_lon_arr), np.nan)
    R_lon_arr = np.full(len(global_lon_arr), np.nan)
    
    # loop over the longitudes to find the latitude in that longitude slice where peak pixel intensity is at 
    for idx, single_lon in enumerate(global_lon_arr):

        # restrict to a longitude slice
        pixel_width = 1
        lon_pixel_spacing = 1
        lon_diff = np.abs(lon_proj - single_lon)
        slice_mask = np.abs(lon_proj - single_lon) <= 1
        #slice_mask = lon_diff <= pixel_width * lon_pixel_spacing

        if np.sum(slice_mask) == 0:
            #print(f"No points found in slice for longitude {single_lon}, skipping...")
            #continue # instead of just skipping the longitude slice altogether, assign NaN for empty slices
            R_peak_lat_arr[idx] = np.nan
            R_lon_arr[idx] = single_lon # still record the longitude
            continue
    
        lon_slice = lon_proj[slice_mask]
        lat_slice = lat_proj[slice_mask]
        rgb_slice = rgb[slice_mask] # needed for plotting later, not for interpolation!
        R_slice = R[slice_mask]
        G_slice = G[slice_mask]
        B_slice = B[slice_mask]

        # save for plotting outside of multithreading
        #interp_slice_dict[(site_name, altitude, time_index, single_lon)] = [lon_slice, lat_slice, rgb_slice]
        interp_slice_arr.append([single_lon, lon_slice, lat_slice, rgb_slice, lon_proj, lat_proj, rgb]) # added og lon_proj, lat_proj, rgb so plotting bounds are nicer


        # flatten yknf and fsmi projected arrays + R + G + B arrays 
        #flattened_points = np.column_stack((lon_proj.flatten(), lat_proj.flatten())) # (N,2) pairs of (lat, lon)
        flattened_points = np.column_stack((lon_slice.flatten(), lat_slice.flatten())) # (N,2) pairs of (lat, lon)
        flattened_R = R_slice.flatten()
        flattened_G = G_slice.flatten()
        flattened_B = B_slice.flatten()

        # remove NaNs
        # masks for NaNs in R, G, and B channels separately (basically remove the NaNs and infinite values if there is) --> where the projected lat, lon, or rgb values corresponding to pixel are NaNs
        nan_mask_R = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(flattened_R))
        nan_mask_G = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(flattened_G))
        nan_mask_B = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(flattened_B))

        points_R_clean = flattened_points[nan_mask_R]
        values_R_clean = flattened_R[nan_mask_R]
        
        points_G_clean = flattened_points[nan_mask_G]
        values_G_clean = flattened_G[nan_mask_G]

        points_B_clean = flattened_points[nan_mask_B]
        values_B_clean = flattened_B[nan_mask_B]

        
        #print(f"{idx}-- Longitude Interpolating: {single_lon}")
        single_lon_arr = single_lon * np.ones(np.shape(lat_arr)) # constant latitude, as many as there are latitudes to interpolate over 
        interp_locations = np.column_stack((np.full_like(lat_arr, single_lon), lat_arr)) # array of (lon, lat) x-y pairs, longitude is constant so get line in lat

        # for Qhull errors 
        if len(points_R_clean) >= 3 and len(values_R_clean >= 3):   # minimum for 2D linear interpolation
            R_intensity_profile = griddata(points_R_clean, values_R_clean, interp_locations, method='linear')
        else:
            R_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        if len(points_G_clean) >= 3 and len(values_G_clean >= 3):   # minimum for 2D linear interpolation
            G_intensity_profile = griddata(points_G_clean, values_G_clean, interp_locations, method='linear')
        else:
            G_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        if len(points_B_clean) >= 3 and len(values_B_clean >= 3):   # minimum for 2D linear interpolation
            B_intensity_profile = griddata(points_B_clean, values_B_clean, interp_locations, method='linear')
        else:
            B_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        # save for plotting outside of multithreading
        #rgb_profile_dict[(site_name, altitude, time_index, single_lon)] = [lat_arr, R_intensity_profile, G_intensity_profile, B_intensity_profile]
        rgb_profile_arr.append([single_lon, lat_arr, R_intensity_profile, G_intensity_profile, B_intensity_profile])
        
        # restrict the latitude to likely latitudes for the aurora (to minimize the outliers)
        # NOT robust for different time stamps 
        # also remove NaNs so curve_fit will work (NaNs added if points to be interpolated outside of convex hull)
        lat_mask = (lat_arr >= lat_min_box - 2) & (lat_arr <= lat_max_box + 2)
        R_nan_mask = np.isfinite(R_intensity_profile)
        R_intensity_restricted = R_intensity_profile[lat_mask & R_nan_mask]
        lat_arr_restricted = lat_arr[lat_mask] 

        if R_intensity_restricted.size==0:
            #print("No points in this latitude range for this longitude slice.\n")
            #continue # add nan instead of skipping all the longitude values
            R_peak_lat_arr[idx] = np.nan
            R_lon_arr[idx] = single_lon # still record the longitude
            continue
        
        # jsut pick out the maximum intensity from the expected range of latitudes
        # indices for intensity & latitude should correspond 
        # make another array for the corresponding longitudes to the peak latitude (accounts for when hitting the exception intensity_restricted.size=0)
        # restrict the longitude so don't get meaningless peak intensities 
        if single_lon <= lon_max_box and single_lon >= lon_min_box:
            idx_peak_R = np.argmax(R_intensity_restricted)
            R_peak_lat_arr[idx] = lat_arr_restricted[idx_peak_R]
        R_lon_arr[idx] = single_lon #always record longitude
            
    # remove outliers in peak latitude before line fitting
    # need to convert to np arrays
    R_peak_lat_arr = np.asarray(R_peak_lat_arr)
    R_lon_arr = np.asarray(R_lon_arr)

    return (interp_slice_arr, rgb_profile_arr,
           R_lon_arr, R_peak_lat_arr)




def reverse_project_lat_lon(lat_target_arr, lon_target_arr, lat_camera, lon_camera, og_h):
    """
        Goal: given lat_target array, lon_target array, find the azimuth and elevation corresponding to them

        Inputs:
            lat_target: global array of latitudes to sample over
            lon_target: the longiutde to sample over for these latitudes
            lat_camera: ground truth camera lat from skymaps
            lon_camera: ground truth camera lon from skymaps
            og_h: height that we projected to in order to get the original / baseline longitudes slice from

        Returns:
            azimuth and elevation arrays
    """
    R = 6371000
    
    # 1. Ensure inputs are numpy arrays
    lat_target_arr = np.atleast_1d(lat_target_arr)
    lon_target_arr = np.atleast_1d(lon_target_arr) # create array same size as lat_target_arr to get all pairs of lat,lon to change back to az,el 

    # Convert to radians bc they were originally in degrees
    lat1, lon1 = np.radians(lat_camera), np.radians(lon_camera)
    lat2, lon2 = np.radians(lat_target_arr), np.radians(lon_target_arr)
    
    dlon = lon2 - lon1

    # 2. Azimuth (vectorized)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    az_rad = np.arctan2(y, x)
    az_deg = np.degrees(az_rad) % 360

    # 3. Central angle phi (vectorized)
    cos_phi = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    cos_phi = np.clip(cos_phi, -1, 1)
    phi = np.arccos(cos_phi)
    
    # 4. Elevation (vectorized)
    # law of cosines to find slant range t
    t_sq = R**2 + (R + og_h)**2 - 2 * R * (R + og_h) * cos_phi
    t_aurora = np.sqrt(np.maximum(t_sq, 0))
    
    # Geometric elevation calc
    # handles the curvature of the Earth correctly for every point in our array
    num = (R + og_h) * cos_phi - R
    # avoid division by zero for points exactly at the observer
    el_rad = np.arcsin(np.divide(num, t_aurora, out=np.zeros_like(num), where=t_aurora!=0))
    el_deg = np.degrees(el_rad)

    return az_deg, el_deg



def plot_lon_slice_bounding_box(lat_proj, lon_proj, 
                                reproj_lat_slice, reproj_lon_slice, #slice
                                reproj_left_lat, reproj_left_lon, reproj_right_lat, reproj_right_lon, #bounding box
                                reproj_bottom_lat, reproj_bottom_lon, reproj_top_lat, reproj_top_lon, #bounding box
                                rgb, time_index, site_name, new_h):
    ''' 
        Plots the projected longiutde slices and the bounding box for a particular projection (sanity check before doing interpolation,
        and to see how much projecting the latitude slices warps the lines and box. 
    '''        
    
    # # Extract time and format it
    # raw_time = yknf_rgb_asi_ds.times.values[time_index]
    # time_obj = pd.to_datetime(raw_time.decode("utf-8").replace(" UTC", ""))
    # time_str = time_obj.strftime("%b. %d, %Y %H:%M:%S UT")
        
    # baseline yknf 110 plot limits, scale larger as projecting to larger altitudes
    # x_plot_min = 221.94 - h_target / 100000 * 25 
    # x_plot_max = 276.13 + h_target / 100000 * 25
    # y_plot_min = 47.37 - h_target / 100000 * 5
    # y_plot_max = 73.27 + h_target / 100000 * 5
    x_plot_min = 215 #lon
    x_plot_max = 275 #lon
    y_plot_min = 45 #lat
    y_plot_max = 80 #lat

    lon_s = np.array(reproj_lon_slice).flatten()
    lat_s = np.array(reproj_lat_slice).flatten()

    # mask out nans for matplotlib 
    mask = np.isfinite(lon_s) & np.isfinite(lat_s)

    # print(reproj_lon_slice)
    # print(reproj_lat_slice)

    # print(lon_s[mask])
    # print(lat_s[mask])

    # x is longitude, y is latitude
    plt.figure(figsize=(8,8))
    plt.scatter(lon_proj.flatten(),lat_proj.flatten(),c=rgb.reshape(-1, 3)/255.0,s=1, alpha=1) #0.15
    if np.any(mask):
        plt.plot(lon_s[mask], lat_s[mask], marker=".", linestyle="-", color='yellow', linewidth=2, label='Slice') # NEED TO FIX THIS BUG, SWITCHED LAT/LON??
    else:
        print("Warning: reproj_slice is entirely NaNs!")
    #plt.plot(reproj_lon_slice, reproj_lat_slice, marker="o", linestyle="-", color='green') 
    plt.plot(reproj_left_lon, reproj_left_lat, marker=".", linestyle="-", color='red')
    plt.plot(reproj_right_lon, reproj_right_lat, marker=".", linestyle="-", color='red')
    plt.plot(reproj_top_lon, reproj_top_lat, marker=".", linestyle="-", color='red')
    plt.plot(reproj_bottom_lon, reproj_bottom_lat, marker=".", linestyle="-", color='red')
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"Overlaid {new_h/1000}km Projection - timeidx{time_index}", pad=30)
    plt.xlim((x_plot_min, x_plot_max))
    plt.ylim((y_plot_min, y_plot_max))
    plt.show()



def new_spherical_project_lat_lon(az_arr, el_arr, lat_camera, lon_camera, new_h):
    '''
    params: 
    az_arr = 2D azimuth array for each pixel (NaNs ok, degrees, xarray)
    el_arr = 2D elevation array for each pixel (NaNs ok, degrees, xarray, no need to be filtered)
    lat_camera = latitude of camera (degrees) --> shape [480, 553]
    lon_camera = longitude of camera (degrees) --> shape [480, 553]
    h = height you want to project azimuth and elevation to to get latitude and longitude for each pixel

    returns: 
    lat_aurora_arr = latitudes of the aurora projected to given height h
    lon_aurora_arr = longitudes of the aurora project to give height h 
    '''

    #----- Preprocessing elevation and azimuth arrays + calculating necessary constants ------#
    # convert to radians + applying mask 
    el_arr = np.radians(np.array(el_arr))
    az_arr = np.radians(np.array(az_arr))
    
    # create elevation mask (True when valid) --> same elevation cutoff as pyaurorax 
    el_mask = (el_arr > np.radians(0.1)) & (el_arr < np.radians(90)) # use a softer mask?
    
    # combine with skymap mask: True = valid pixel
    valid_mask = el_mask #& (~skymap_110_mask) # shouldnt need this skymap110 masking; elevation clipping should be sufficient --> if things still look bad without it, then something else is wrong 
    
    # set invalid pixels to NaN
    el_arr[~valid_mask] = np.nan

    # distance along ray to intersect with aurora circle
    R = 6371000 # avg radius of earth 
    t_aurora = -1*R*np.sin(el_arr) + np.sqrt(R**2*(np.sin(el_arr))**2 + 2*new_h*R + new_h**2) # this gives the world point P!

    # x component of the ray when it hits the aurora circle (head-on 2D perspective)
    x_aurora = t_aurora * np.cos(el_arr)

    # angle giving the arc distance of lat/lon offset from the camera 
    phi = np.arcsin(x_aurora / (R+new_h))

    # arc distance of the lat/long offset from the camera (needs to be decomposed)
    s = R * phi

    # horizontal distance between camera and aurora along tangent plane approximation to earth
    dx_arr = s * np.sin(az_arr) # dist along camera plane east (m)
    dy_arr = s * np.cos(az_arr) # dist along camera plane north (m)

    # convert N/E offset components to (lat, lon) --> comes out in decimal degrees 
    lat_delta_arr = dy_arr / 111045 #degrees
    lon_delta_arr = dx_arr / (np.cos(np.radians(lat_camera + lat_delta_arr)) * 111321) 
    #lat_delta_arr = dx_arr / 111045 #degrees
    #lon_delta_arr = dy_arr / (np.cos(np.radians(lat_camera + lat_delta_arr)) * 111321) 

    # add lat/long offset to camera's og lat/lon to get the lat/lon of the aurora at the chosen height!
    lat_aurora_arr = lat_camera + lat_delta_arr
    lon_aurora_arr = lon_camera + lon_delta_arr

    # apply the same mask again just in case --> removed for now for testing
    #mask_restricted = skymap_110_mask[1:, 1:]
    #lat_aurora_arr[skymap_110_mask] = np.nan
    #lon_aurora_arr[skymap_110_mask] = np.nan

    # print(f"DEBUG: Mean Lat is {np.nanmean(lat_aurora_arr)}") # should be ~60-70
    # print(f"DEBUG: Mean Lon is {np.nanmean(lon_aurora_arr)}") # should be ~220-250


    return lat_aurora_arr, lon_aurora_arr



def project_lat_slices_and_box(lat_slice_target_arr, lon_slice_target_arr, 
                               lat_box_max, lat_box_min, lon_box_max, lon_box_min,
                               lat_camera, lon_camera, og_h, new_h):
    """
    Goal: loop over all of the longitudes in lon_target_arr, match each lon with a latitude in lat_target array to get lat,lon
          pairs for 1 longitude slice. apply reverse_project_lat_lon() in order to get the azimuth and elevation for each lat,lon
          (still inside the loop for 1 longitude). then reproject the azimuth and elevation back to lat,lon but to a new height.
          then in fixed_line_interpolate(), need to take these arrays of lat,lon projected to a new height and put them into the 
          interpolation function!


          ** these are all in degrees **
          lat_slice_target_arr = all the different latitudes for which to look at for each longitude slice (so constant)
          lon_slice_target_arrr = array of all the different lontiudes to slice the aurora at
          single_lon_target_arr = taking one of the longitudes from lon_slice_target_arr and making an array same size as lat_slice_target_arr of all the same lon value
          

    output:
        reproj_lat_arr_dict = dictionary w/ key: the original longitude slice we were sampling at (from og_h), value: array of lats after the new spherical projection (to new_h)
        reproj_lon_arr_dict = dictionary w/ key: the original longitude slice we were sampling at (from og_h), value: array of lats after the new spherical projection (to new_h)
        reproj_left_arr = array of (lat, lon) pairs of the original left side of the bounding box projected up to new_h
        reproj_right_arr = array of (lat, lon) pairs of the original right side of the bounding box projected up to new_h
        reproj_bottom_arr = array of (lat, lon) pairs of the original bottom side of the bounding box projected up to new_h
        reproj_top_arr = array of (lat, lon) pairs of the original top side of the bounding box projected up to new_h
        ^ now instead of doing the pairs, need to return the raw latitude and longitude for each edge of the bounding box --> 2 + 4*2=10 total return values

    """
    #print(f"BEGIN lat_slice_target_arr in project_lat_slices_and_box: {lat_slice_target_arr}")
    #print(f"BEGIN lon_slice_target_arr in project_lat_slices_and_box: {lon_slice_target_arr}")
    

    # reprojecting the longitude slices we are sampling at
    reproj_lat_arr_dict = {}
    reproj_lon_arr_dict = {}
    for lon_slice_target in lon_slice_target_arr:
        single_lon_target_arr = np.full(shape=len(lat_slice_target_arr), fill_value=lon_slice_target)
        reproj_az_arr, reproj_el_arr = reverse_project_lat_lon(lat_slice_target_arr, single_lon_target_arr, lat_camera, lon_camera, og_h)
        reproj_lat_arr, reproj_lon_arr = new_spherical_project_lat_lon(reproj_az_arr, reproj_el_arr, lat_camera, lon_camera, new_h)
        reproj_lat_arr_dict[lon_slice_target] = reproj_lat_arr
        reproj_lon_arr_dict[lon_slice_target] = reproj_lon_arr

        #print(f"{lon_slice_target}: reproj lat arr: {reproj_lat_arr}")
        #print(f"{lon_slice_target}: reproj lon arr: {reproj_lon_arr}")

    # reverse project lat/lon expect degrees 

    # reprojecting the bounding box (separately loop over the 2 longtiudes and 2 latitudes)
    lat_box_arr = np.arange(lat_box_min, lat_box_max, 0.5) # need to correpond to to the lon_min_box and lon_max_box (2 separate arrays with lon cst), maybe change step
    lon_box_arr = np.arange(lon_box_min, lon_box_max, 0.5) # need to correspond to the lat_min_box and lat_max_box (2 separate arrays with lat cst), maybe change step
    
    # over latitudes
    lon_box_min_arr = np.full(shape=len(lat_box_arr), fill_value=lon_box_min) #cst value lonmin
    lon_box_max_arr = np.full(shape=len(lat_box_arr), fill_value=lon_box_max) #cst value lonmax
    az_reproj_left_top_to_bottom, el_reproj_left_top_to_bottom = reverse_project_lat_lon(lat_box_arr, lon_box_min_arr, lat_camera, lon_camera, og_h) #varying lat at lonmin
    az_reproj_right_top_to_bottom, el_reproj_right_top_to_bottom = reverse_project_lat_lon(lat_box_arr, lon_box_max_arr, lat_camera, lon_camera, og_h) #varying lat at lonmax
    reproj_left_arr_lat, reproj_left_arr_lon = new_spherical_project_lat_lon(az_reproj_left_top_to_bottom, el_reproj_left_top_to_bottom, lat_camera, lon_camera, new_h)
    reproj_right_arr_lat, reproj_right_arr_lon = new_spherical_project_lat_lon(az_reproj_right_top_to_bottom, el_reproj_right_top_to_bottom, lat_camera, lon_camera, new_h)
    
    # over longitudes  
    lat_box_min_arr = np.full(shape=len(lon_box_arr), fill_value=lat_box_min) #cst value latmin
    lat_box_max_arr = np.full(shape=len(lon_box_arr), fill_value=lat_box_max) #cst value latmax
    az_reproj_bottom_left_to_right, el_reproj_bottom_left_to_right = reverse_project_lat_lon(lat_box_min_arr, lon_box_arr, lat_camera, lon_camera, og_h) #varying lon at latmax
    az_reproj_top_left_to_right, el_reproj_top_left_to_right = reverse_project_lat_lon(lat_box_max_arr, lon_box_arr, lat_camera, lon_camera, og_h) #varying lon at latmin
    reproj_bottom_arr_lat, reproj_bottom_arr_lon = new_spherical_project_lat_lon(az_reproj_bottom_left_to_right, el_reproj_bottom_left_to_right, lat_camera, lon_camera, new_h)
    reproj_top_arr_lat, reproj_top_arr_lon = new_spherical_project_lat_lon(az_reproj_top_left_to_right, el_reproj_top_left_to_right, lat_camera, lon_camera, new_h)

    # inside project_lat_slices_and_box, after reverse_project
    #print(f"Elevation angles range: {np.nanmin(az_reproj_left_top_to_bottom)} to {np.nanmax(el_reproj_left_top_to_bottom)}")

    
    return (reproj_lat_arr_dict, reproj_lon_arr_dict,
            reproj_left_arr_lat, reproj_left_arr_lon, 
            reproj_right_arr_lat, reproj_right_arr_lon, 
            reproj_bottom_arr_lat, reproj_bottom_arr_lon,
            reproj_top_arr_lat, reproj_top_arr_lon)




''' fixing to get consistent longitude array size --> pick same number of longitudes to interpolate over AND also added fixed the projecting longitudes + bounding box from 2-26'''
def fixed_line_interpolate(lat_proj, lon_proj, rgb, 
                             lat_min_box, lat_max_box, 
                             lon_min_box, lon_max_box,
                             lat_camera, lon_camera,
                             site_name, time_index, 
                             og_h, new_h,
                             global_lon_arr, global_lat_arr):

    """
    Goal:
    Given the inputted lat_proj, lon_proj arrays (the latitude and longitude projected to height new_h). We want to sample along a line "line"
    in longitude, so over an array of lat,lon pairs. So we predefine our points of longitude over which to slice (relative to our "baseline" projection of 150km)
    and get an 0.5-spaced latitude array that go across the latitude line for 150km (this is the true line since this is the baseline; any 
    projections will curve the line a bit. The pairs here will look like (lat, cst lon) for each lon slice. 
    Add a condition, if new_h is 150km, skip the reprojection, else reproject the latitude lines and the 4 edges of the bounding box. The
    lat/lon min/max for the bounding box is now defined from the baseline 150km projection, and projected upwards, like the latitude slices. 
    So, we will be getting the array of (lat,lon) pairs to interpolate over (interp_locations) from the reproj_lat_arr_dict and the 
    reproj_lon_arr_dict (this defines around which line to interpolate). Now we need the points (lat,lon) and values (r channel) over which 
    to interpolate. For the points, need a slice that is wide enough to capture the entire line (choose an arbitrary length for now, ie mask
    the lat_proj and lon_proj arrays with threshold as long as within +-2 degrees of longitude, include it in the slice, maybe need to make 
    this larger threshold depending on how much the longitude slice line curves when projected upwards (make plots for this and the bounding
    box). For the values, need a slice of the r-channel array that follows where the lat/lons were zeroed, so just apply the same slice mask
    to the r-channel array. Now, with the interp_locations, points, and values, can pass into graddata to interpolate, and it will spit out
    The output of griddata/interpolation is an array same size as interp_locations, with each element the brightness of pixel
    corresponding to each (lat,lon) pairs passed in as interp_locations. Now restrict this array to the projected bounding box (polygon).
    Note that the bounding box is common across yknf and fsmi. 
    
    input:
        lat_proj = 2D projected latitude arr (output of project_lat_lon())
        lon_proj = 2D projected longitude arr (output of project_lat_lon())
        rgb = 3D rgb array (output of mod_plot_lat_lon()) (these are the raw rgb values from the skymap, need to be matched to the projected new projected lat/lon
        lat_min_box = restricted latitude for where aurora is in the frame (needed for finding max intensity) for 130km projection (lower bound for where we are getting the metric matrix)
        lat_max_box = restricted latitude for where aurora is in the frame (needed for finding max intensity) for 130km projection (lower bound for where we are getting the metric matrix)
        lon_min_box = restrict longitude for where aurora is in the frame (else will get meaningless peak intensities and screw up line of best fit)
        lon_max_box = restrict longitude for where aurora is in the frame (else will get meaningless peak intensities and screw up line of best fit)
        time_index = for naming purposes when plotting 
        site_name = for naming purposes when plotting
        og_h = 150,000 km the baseline for where we determined the best longitude slices and the best lat/lon box
        new_h = new height we've projected to for the interpolation (what lat_proj and lon_proj were projected to) 

    output:
        R_peak_lat_arr = 1D arr of all the latitudes that the R-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        G_peak_lat_arr = 1D arr of all the latitudes that the G-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        B_peak_lat_arr = 1D arr of all the latitudes that the B-channel had peak altitude for, interpolated over the different longitudes + restricted to (lat_min_box, lat_max_box)
        lon_arr = 1D arr of all the longitudes corresponding in idx to the peak latitude arrays (should be same length as the other 3 returned arrays)
    """
   # print(f"ALTITUDE: {new_h}")
    
    # latitude and longitude arrays to define the interpolation grid
    #lat_slice_target_arr = np.arange(lat_min, lat_max, 0.5) # this is about close to the actual pixel resolution --> these are in degrees
    #lon_slice_target_arr = np.arange(lon_min, lon_max, 1) # resolution chosen arbitrarily hmmm

    # should be the same amount before and after reprojection! -->  need to pick a new bounding box for the lowest projection going to for the matrix (130km)
    # in reality it's not really the "longitudes" we are interpolating at, except at 130km --> projecting upwards we are actually interpolating at a curve! 
    print(f"\n====={new_h/1000.0}km PROJECTION=======")
    print(f"Total {len(global_lon_arr)} longitudes, and for each of these longitudes have   {len(global_lat_arr)} latitudes to interpolate\n")

    # separate into R channel
    R = rgb[:,:, 0]

    # preallocate the sizes of the R_peak_lat_array and R_lon_arr (don't <continue> when run into issue with longitude, rather just add nan in that place)
    R_peak_lat_arr = np.full(len(global_lon_arr), np.nan)
    R_lon_arr = np.full(len(global_lon_arr), np.nan)

 #   if og_h != new_h: --> maybe need to differentiate this, bc if projecting to 130km then it will just be a line 
    # reproject the latitude slices and the bounding box for each projection --> already looping through all the lon slices here
    reproj_lat_arr_dict, reproj_lon_arr_dict, reproj_left_lat, reproj_left_lon, reproj_right_lat, reproj_right_lon, reproj_bottom_lat, reproj_bottom_lon, reproj_top_lat, reproj_top_lon = project_lat_slices_and_box(global_lat_arr, global_lon_arr, # these are in degrees
                                                                                                                                                                                                                           lat_max_box, lat_min_box, lon_max_box, lon_min_box, # also degrees
                                                                                                                                                                                                                           lat_camera, lon_camera, og_h, new_h)
    # LEAVE THIS WHITESPACE ALONE
    original_lon_arr = reproj_lat_arr_dict.keys() # these keys should be the same as take from reproj_lon_arr_dict, and in degrees
   # print(f"ORIGINAL LON ARR: {original_lon_arr}")
    reproj_lon_slice_arr = reproj_lon_arr_dict.values() # this is array of arrays of longitude corresp. to each longitude of the original slices
    reproj_lat_slice_arr = reproj_lat_arr_dict.values()
   # print(f"{site_name}{new_h} reproj_lat_slice_arr len: {len(reproj_lat_slice_arr)}")
  #  print(f"REPROJ LAT SLICE ARR: {reproj_lat_slice_arr}")

    # define bounding box as a polygon, use for masking later 
    box_lats = np.concatenate([reproj_top_lat, reproj_right_lat, reproj_bottom_lat, reproj_left_lat])
    box_lons = np.concatenate([reproj_top_lon, reproj_right_lon, reproj_bottom_lon, reproj_left_lon])
    vertices = np.column_stack((box_lats, box_lons))
    box_boundary_path = Path(vertices)

    # preallocate
    num_slices = len(original_lon_arr)
    R_peak_lat_arr = np.full(num_slices, np.nan)
    R_lon_arr = np.full(num_slices, np.nan)# longitudes that correspond to the peak latitude 
    for idx, (original_lon,reproj_lat_slice, reproj_lon_slice) in enumerate(zip(original_lon_arr, reproj_lat_slice_arr, reproj_lon_slice_arr)):

        # # plot the reprojected longitude slice line and the reprojected bounding box
        # plot_lon_slice_bounding_box(lat_proj, lon_proj, 
        #                             reproj_lat_slice, reproj_lon_slice, #slice
        #                             reproj_left_lat, reproj_left_lon, reproj_right_lat, reproj_right_lon, #bounding box
        #                             reproj_bottom_lat, reproj_bottom_lon, reproj_top_lat, reproj_top_lon, #bounding box
        #                             rgb, time_index, site_name, new_h)
        lon_buffer = 10 + (new_h / 100000)
        slice_mask = np.abs(lon_proj - original_lon) <= lon_buffer # might need to change this threshold depending on warp in projection--> had to change it to be a LOT larger! or else points not showing up 
        lon_slice_points = lon_proj[slice_mask]
        lat_slice_points = lat_proj[slice_mask]
        R_slice_values = R[slice_mask] #already flattened 

        # get lat and lon slices over which to interpolate in the right shape 
        flattened_points = np.column_stack((lat_slice_points.flatten(), lon_slice_points.flatten())) # (N,2) pairs of (lat, lon)

        # masks for NaNs in R, G, and B channels separately (basically remove the NaNs and infinite values if there is) --> where the projected lat, lon, or rgb values corresponding to pixel are NaNs
        nan_mask_R = (np.isfinite(flattened_points[:,0]) & np.isfinite(flattened_points[:,1]) & np.isfinite(R_slice_values))
        points_R_clean = flattened_points[nan_mask_R]
        values_R_clean = R_slice_values[nan_mask_R]
        #print(f"Lon {original_lon}: Points captured by mask: {len(points_R_clean)}")

        # get interpolation locations (these are just the (lat,lon) pairs from the reprojected longitude slices
        interp_locations = np.column_stack((reproj_lat_slice, reproj_lon_slice))

        # print("REPROJ LAT SLICE")
        # print(reproj_lat_slice)
        # print("REPROJ LON SLICE")
        # print(reproj_lon_slice)

        if len(points_R_clean) >= 3 and len(values_R_clean) >= 3:   # minimum for 2D linear interpolation
            R_intensity_profile = griddata(points_R_clean, values_R_clean, interp_locations, method='linear') #intensities along the interp_locations
        else:
            R_intensity_profile = np.full(len(interp_locations), np.nan)  # or 0

        valid_interp_count = np.sum(np.isfinite(R_intensity_profile))
        print(f"Lon {original_lon}: Points successfully interpolated: {valid_interp_count}")

        
        bounding_box_mask = box_boundary_path.contains_points(interp_locations) #expects (N,2) like griddata 
        R_nan_mask = np.isfinite(R_intensity_profile)
        R_intensity_restricted = R_intensity_profile[bounding_box_mask & R_nan_mask]
        reproj_lat_slice_restricted = reproj_lat_slice[bounding_box_mask & R_nan_mask]
        reproj_lon_slice_restricted = reproj_lon_slice[bounding_box_mask & R_nan_mask]

        points_in_box = np.sum(bounding_box_mask)
        print(f"Lon {original_lon}: Points inside bounding box: {points_in_box}")

        if R_intensity_restricted.size==0:
            #print("No points in this latitude range for this longitude slice.\n")
            #continue # add nan instead of skipping all the longitude values
            R_peak_lat_arr[idx] = np.nan
            R_lon_arr[idx] = original_lon # still record the longitude, tho this is kinda meaningless bc it is the pre-projection lon
            continue

        # along each latitude "line", pick out the peak latitude
        index_of_peak_intensity = np.argmax(R_intensity_restricted)
        peak_latitude = reproj_lat_slice_restricted[index_of_peak_intensity]
        lon_of_peak_lat = reproj_lon_slice_restricted[index_of_peak_intensity]
        R_peak_lat_arr[idx] = peak_latitude
        R_lon_arr[idx] = lon_of_peak_lat

    # need to convert to np arrays
    R_peak_lat_arr = np.asarray(R_peak_lat_arr)
    R_lon_arr = np.asarray(R_lon_arr)

    print(f"\n{site_name}{new_h}m: len of R_peak_lat_arr:{len(R_peak_lat_arr)}, len of R_lon_arr:{len(R_lon_arr)}")

    return R_lon_arr, R_peak_lat_arr

def new_compute_metrics_for_altitude(
    og_h, new_h,
    t_arr,
    full_azimuth_yknf, full_elevation_yknf, lat_cam_yknf, lon_cam_yknf,
    full_azimuth_fsmi, full_elevation_fsmi, lat_cam_fsmi, lon_cam_fsmi,
    yknf_rgb_asi_ds, fsmi_rgb_asi_ds,
    global_lon_arr, global_lat_arr# so same number of longitude slices for all the different projected altitudes
):
    # ---- project ONCE per altitude ----
    yknf_lat_proj, yknf_lon_proj = new_spherical_project_lat_lon(
        full_azimuth_yknf, full_elevation_yknf,
        lat_cam_yknf, lon_cam_yknf, new_h
    )

    fsmi_lat_proj, fsmi_lon_proj = new_spherical_project_lat_lon(
        full_azimuth_fsmi, full_elevation_fsmi,
        lat_cam_fsmi, lon_cam_fsmi, new_h
    )

    # #---------Define a single global longitude grid -------------+
    # global_lon_min = int(np.floor(
    #     min(np.nanmin(yknf_lon_proj), np.nanmin(fsmi_lon_proj))
    # ))
    # global_lon_max = int(np.ceil(
    #     max(np.nanmax(yknf_lon_proj), np.nanmax(fsmi_lon_proj))
    # ))

    # # ---- overlap region (once) ----
    # lat_min = min(np.nanmin(yknf_lat_proj), np.nanmin(fsmi_lat_proj))
    # lon_min = min(np.nanmin(yknf_lon_proj), np.nanmin(fsmi_lon_proj))
    # lat_max = max(np.nanmax(yknf_lat_proj), np.nanmax(fsmi_lat_proj))
    # lon_max = max(np.nanmax(yknf_lon_proj), np.nanmax(fsmi_lon_proj))

    # fixed auroral box this is based on where the aurora falls for yknf and fsmi relative to our baseline of 150km
    # will be projecting this entire box up 
    # already have a fixed global_lon_arr and global_lat_arr that we are projecting up each time 
    lat_min_box, lat_max_box = 60.5, 66
    lon_min_box, lon_max_box = 240, 260

    ssd_vals = [] # sum of squared differences
    sad_vals = [] # sum of absolute differences
    avg_diffs = [] # average of absolute differences
    med_diffs = [] # median of absolute differences 
    diffs = []

    yknf_peaks_all = []  # store peak arrays for each time_index
    fsmi_peaks_all = []
    
    for time_index in t_arr:
        # ---- extract RGB frames ----
        yknf_rgb, fsmi_rgb = mod_plot_lat_lon(
            yknf_rgb_asi_ds, fsmi_rgb_asi_ds,
            time_index,
            "Yellowknife", "Fort Smith",
            yknf_lat_proj, yknf_lon_proj,
            fsmi_lat_proj, fsmi_lon_proj,
            new_h
        )
        

        # ---- interpolate (only need peak arrays) ----
        _, yknf_peak = fixed_line_interpolate(
            yknf_lat_proj, yknf_lon_proj, yknf_rgb,
             lat_min_box, lat_max_box, 
             lon_min_box, lon_max_box,
             lat_cam_yknf, lon_cam_yknf,
             "YKNF", time_index, 
             og_h, new_h,
             global_lon_arr, global_lat_arr        
        )

        _, fsmi_peak = fixed_line_interpolate(
             fsmi_lat_proj, fsmi_lon_proj, fsmi_rgb, 
             lat_min_box, lat_max_box, 
             lon_min_box, lon_max_box,
             lat_cam_fsmi, lon_cam_fsmi,
             "FSMI", time_index, 
             og_h, new_h,
             global_lon_arr, global_lat_arr
        )

        yknf_peaks_all.append(yknf_peak)
        fsmi_peaks_all.append(fsmi_peak)

    # convert to arrays 
    yknf_peaks_all = np.array(yknf_peaks_all)
    fsmi_peaks_all = np.array(fsmi_peaks_all)

    global_valid_mask = np.isfinite(yknf_peaks_all) & np.isfinite(fsmi_peaks_all)

    for t_idx in range(len(t_arr)):
        diff = yknf_peaks_all[t_idx][global_valid_mask[t_idx]] - fsmi_peaks_all[t_idx][global_valid_mask[t_idx]]
       
        # ---- Metrics ----
        ssd_vals.append(np.nansum(np.abs(diff**2)))
        sad_vals.append(np.nansum(np.abs(diff)))
        avg_diffs.append(np.mean(np.abs(diff)))
        med_diffs.append(np.median(np.abs(diff)))
        diffs.append(np.abs(diff)) # 2d array of the differences for investigation 

    return new_h, ssd_vals, sad_vals, avg_diffs, med_diffs, diffs



