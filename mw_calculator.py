#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:32:03 2022

@author : ARHAM ZAKKI EDELO
@contact: edelo.arham@gmail.com
"""

import os, glob, subprocess, sys, warnings
from pathlib import Path, PurePath
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from loguru import logger
from obspy import UTCDateTime, Stream, read, Trace, read_inventory
from obspy.signal import rotate
from obspy.geodetics import gps2dist_azimuth

import refraction as ref
import fitting_spectral as fit

warnings.filterwarnings('ignore')

print('''
Python code to calculate moment magnitude

Before you run this program, make sure you have changed all the path correctly.      
      ''')

# Global Parameters
#============================================================================================

# instrument correction parameters
WATER_LEVEL = 0                      # water level 
PRE_FILTER = [0.1, 0.75, 29.25, 30] # these values need to be customized for a specific bandwidth

# plotting parameters
# static window parameter 
#time_after_pick_p = 0.45
#time_after_pick_s = 1.2

# time padding and position parameters
TIME_BEFORE_PICK = 0.1

# noise spectra duration before P pick
NOISE_DURATION = 0.5
NOISE_PADDING = 0.2

# setting frequency range for spectral fitting (default spectral f-band frequency range)
F_MIN = 0.75
F_MAX = 25


def start_calculate(
        wave_path: Path,
        hypo_path: Path,
        sta_path: Path, 
        pick_path: Path, 
        cal_path: Path,
        fig_path: Path
        ) ->  Tuple [pd.DataFrame, pd.DataFrame, str]:

    """
    Start the process of moment magnitude calculation.
    
    Args:
        wave_path (Path): Path to the waveforms file.
        hypo_path (Path): Path to the hypocenter catalog file (.excel format).
        sta_path (Path) : Path to the station file (.excel format).
        pick_path (Path): Path to the catalog picking file (.excel format).
        cal_path (Path): Path to the calibration file (.RESP format).
        fig_path (Path): Path to the directory where the image of the result of spectral fitting will be stored.
        
    Returns:
        Tuple [pd.Dataframe, pd.DataFrame, str]: DataFrames for magnitude results and fitting results, and the output file name.
    """
        
    prompt=str(input('Please type yes/no if you had changed the path :'))
    if prompt != 'yes':
        sys.exit("Ok, please correct the path first!")
    else:
        print("Process the program ....\n\n")
        pass

    # setting logger for debugging 
    logger.remove()
    logger.add("runtime.log", level="ERROR", backtrace=True, diagnose=True)
    
    # Get the user input.
    id_start, id_end, mw_output, fig_state = get_user_input()
    
    # loading file input
    hypo_data    = pd.read_excel(hypo_path, index_col = None) 
    pick_data    = pd.read_excel(pick_path, index_col = None)
    station_data = pd.read_excel(sta_path, index_col = None)

    # initiate dataframe for magnitude calculation results
    df_result   = pd.DataFrame(
                        columns = ["ID", "Fc", "Fc_std", "Mw", "Mw_std", "Src_rad(m)", "Src_rad_std", "Stress_drop(bar)", "Stress_drop_std"] 
                        )
    df_fitting  = pd.DataFrame(
                        columns = ["ID", "Station", "F_corner_P", "F_corner_SV", "F_corner_SH", "Qfactor_P", "Qfactor_SV", "Qfactor_SH", "Omega_0_P(nms)", "Omega_0_SV(nms)",  "Omega_0_SH(nms)", "RMS_e_P(nms)", "RMS_e_SV(nms)", "RMS_e_SH(nms)"] 
                        )

    for _id in range (id_start, id_end + 1):
        
        print(f"Calculate moment magnitude for event ID {_id} ...")
        
        # get the dataframe 
        hypo_data_handler   = hypo_data[hypo_data["ID"] == _id]
        pick_data_handler   = pick_data[pick_data["Event ID"] == _id]

        # start calculating moment magnitude
        try:
            # calculate the moment magnitude
            mw_results, fitting_result = calculate_moment_magnitude(wave_path, hypo_data_handler, pick_data_handler, station_data, cal_path, _id, fig_path, fig_state)

            # create the dataframe from calculate_ml_magnitude results
            mw_magnitude_result = pd.DataFrame.from_dict(mw_results)
            mw_fitting_result   = pd.DataFrame.from_dict(fitting_result)
            
            # concatinate the dataframe
            df_result = pd.concat([df_result, mw_magnitude_result], ignore_index = True)
            df_fitting = pd.concat([df_fitting, mw_fitting_result], ignore_index = True)
            
        except Exception as e:
            logger.error(f"An error occured during calculation for event {_id}: {e}")
            pass
    
    return df_result, df_fitting, mw_output



def get_user_input () -> Tuple[int, int, str, bool]:
    """
    Get user inputs for processing parameters.
    
    Returns:
        Tuple[int, int, str, bool]: Start ID, end ID, output name, and whether to generate figures.
    """
    
    id_start    = int(input("Event ID to start the moment magnitude calculation : "))
    id_end      = int(input("Event ID to end the moment magnitude calculation : "))
    mw_output   = str(input("Result file name? (ex. mw_out): "))
    fig_state   = input("Do you want to produce the spectral fitting image [yes/no]?: ")
    
    # check status image builder
    if fig_state == 'yes':
        fig_state = True
    else:
        fig_state = False
        
    return id_start, id_end, mw_output, fig_state



def read_waveforms(path: Path, event_id: str) -> Stream:
    """
    Read waveforms file (.mseed) from the specified path and event id.

    Args:
        path (Path): Parent path of separated by id waveforms directory.
        event_id (float): Unique identifier for the earthquake event.

    Returns:
        Stream: A Stream object containing all the waveforms from specific event id.
    """
    
    stream = Stream()
    for w in glob.glob(os.path.join(path.joinpath(f"{event_id}"), '*.mseed'), recursive = True):
        try:
            stread = read(w)
            stream += stread
        except Exception as e:
            logger.error(f"Error reading waveform {w} for event {event_id}: {e} ")
    
    return stream



def instrument_remove (st: Stream, calibration_path: Path, fig_path: Optional[str] = None, fig_statement: bool = False) -> Stream:
    """
    Removes instrument response from a Stream of seismic traces using calibration files.

    Args:
        st (Stream): A Stream object containing seismic traces with instrument responses to be removed.
        calibration_path (str): Path to the directory containing the calibration files in RESP format.
        fig_path (Optional[str]): Directory path where response removal plots will be saved. If None, plots are not saved.
        fig_statement (bool): If True, saves plots of the response removal process. Defaults to False.

    Returns:
        Stream: A Stream object containing traces with instrument responses removed.
    """
    st_removed=Stream()
    
    for tr in st:
        try:
            # Construct the calibration file
            sta, comp = tr.stats.station, tr.stats.component
            inv_path = calibration_path.joinpath(f"RESP.KJ.{sta}..BH{comp}")
            
            # Read the calibration file
            inv = read_inventory(inv_path, format='RESP')
  
            # Prepare plot path if fig_statement is True
            plot_path = None
            if fig_statement and fig_path:
                plot_path = fig_path.joinpath(f"fig_{sta}_BH{comp}")
            
            # Remove instrument response
            rtr = tr.remove_response(
            inventory = inv,
            pre_filt = PRE_FILTER,
            water_level = WATER_LEVEL,
            output = 'DISP',
            zero_mean = True,
            taper = True,
            taper_fraction = 0.05,
            plot = plot_path
            )

            # Re-detrend the trace
            rtr.detrend("linear")
            
            # Add the processed trace to the Stream
            st_removed+=rtr
            
        except Exception as e:
            logger.error(f"Error process instrument removal in trace {tr.id}: {e}")
            continue
            
    return st_removed
    
    
    
def rotate_component(st: Stream, azim: float, inc: float) -> Stream  :
    """
    Rotates a stream of seismic traces from the ZNE (Vertical-North-East) component system
    to the LQT (Longitudinal-Transverse-Vertical) component system based on given azimuth
    and inciddence angles.

    Args:
        st (Stream): A Stream object containing the Z, N, and E components as traces.
        azim (float): The azimuth angle (in degrees) for rotation.
        inc (float): The inciddence angle (in degrees) for rotation.

    Returns:
        Stream: A Stream object containing the rotated L, Q, and T components as traces.
    """
    
    # Create an empty Stream object to hold the rotated traces
    sst_rotated = Stream()
    
    # Extract the traces and their metadata
    tr_Z = st.select(component='Z')[0]
    tr_L_status = tr_Z.stats
    
    tr_N = st.select(component='N')[0]
    tr_T_status = tr_N.stats
    
    tr_E = st.select(component='E')[0]
    tr_Q_status = tr_E.stats
    
    # Perform the rotation using the provided azimuth and inclination
    tr_L_data, tr_Q_data, tr_T_data = rotate.rotate_zne_lqt(tr_Z.data, tr_N.data, tr_E.data, azim, inc)
    
    # Convert numpy ndarrays to Trace objects and update their metadata
    tr_L = Trace(tr_L_data, header = tr_L_status)
    tr_L.stats.component = 'L'
    tr_Q = Trace(tr_Q_data, header = tr_Q_status)
    tr_Q.stats.component = 'Q'
    tr_T = Trace(tr_T_data, header = tr_T_status)
    tr_T.stats.component = 'T'

    # Add the rotated traces to the Stream
    sst_rotated.extend([tr_L, tr_Q, tr_T])
    return sst_rotated
    
    
    
def window_trace(tr: Trace, P_arr: float, S_arr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Windows seismic trace data around P, SV, and SH phase and extracts noise data.

    Args:
        tr (Trace): A Trace object containing the seismic data.
        P_arr (float): The arrival time of the P phase (in seconds from the trace start).
        S_arr (float): The arrival time of the S phase (in seconds from the trace start).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - P_data: The data windowed around the P phase in the L component.
            - SV_data: The data windowed around the S phase in the Q component.
            - SH_data: The data windowed around the S phase in the T component.
            - noise_data: The data windowed around the noise period before the P phase in the L component.
    """
    # Extract the vertical, radial, and transverse components
    tr_L = tr.select(component='L')[0]
    [tr_Q, tr_T] = [tr.select(component = comp)[0] for comp in ['Q', 'T']]
    
    # Dynamic window parameters
    s_p_time = float(S_arr - P_arr)    
    time_after_pick_p = 0.75 * s_p_time
    time_after_pick_s = 1.50 * s_p_time
    
    # Calculate indices for P phase windowing
    p_phase_start_index = int(round( (P_arr - tr_L.stats.starttime )/ tr_L.stats.delta,4)) - \
                            int(round(TIME_BEFORE_PICK  / tr_L.stats.delta,4))
    p_phase_finish_index = int(round((P_arr - tr_L.stats.starttime )/ tr_L.stats.delta,4))+ \
                            int(round(time_after_pick_p / tr_L.stats.delta,4))
                            
    P_data     = tr_L.data[p_phase_start_index : p_phase_finish_index + 1]
    
    # Calculate indices for SV and SH phase windowing
    s_phase_start_index = int(round( (S_arr - tr_Q.stats.starttime )/ tr_Q.stats.delta,4))- \
                            int(round(TIME_BEFORE_PICK / tr_Q.stats.delta,4))
    s_phase_finish_index = int(round((S_arr - tr_Q.stats.starttime )/ tr_Q.stats.delta,4))+ \
                            int(round(time_after_pick_s / tr_Q.stats.delta,4))

    SV_data     = tr_Q.data[s_phase_start_index : s_phase_finish_index + 1]
    SH_data     = tr_T.data[s_phase_start_index : s_phase_finish_index + 1]
    
    # Calculate indices for noise data windowing
    noise_start_index = int(round( (P_arr - tr_L.stats.starttime )/ tr_L.stats.delta,4)) - \
                            int(round( NOISE_DURATION / tr_L.stats.delta,4))
    noise_finish_index  = int(round( (P_arr - tr_L.stats.starttime )/ tr_L.stats.delta,4)) - \
                            int(round( NOISE_PADDING / tr_L.stats.delta,4))

    noise_data = tr_L.data[noise_start_index : noise_finish_index + 1]

    return P_data, SV_data, SH_data, noise_data



def trace_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) based on the RMS (Root Mean Square) of the signal and noise.

    Args:
        data (np.ndarray): Array of signal data.
        noise (np.ndarray): Array of noise data.

    Returns:
        float: The Signal-to-Noise Ratio (SNR), calculated as the ratio of the RMS of the signal to the RMS of the noise.
    """
    
    # Compute RMS of the signal
    data_rms = np.sqrt(np.mean(np.square(data)))
    
    # Compute RMS of the noise
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    
    return data_rms / noise_rms



def calculate_spectra(trace_data: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Power Spectral DENSITY (PSD) of a given signal using the Welch method.

    Args:
        trace_data (np.ndarray): Array of signal data to analyze.
        sampling_rate (float): Sampling rate of the signal in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - frequency: Array of sample frequencies.
            - power_spectra: Array of power spectral densities corresponding to the frequencies.
    """
    
    # Ensure the input data is a numpy array
    trace_data = np.asarray(trace_data)
    
    # Check if the trace_data is non-empty
    if len(trace_data) == 0:
        raise ValueError("trace_data cannot be empty.")
    
    # Calculate Power Spectral DENSITY using Welch's method
    frequency, power_spectra = signal.welch(trace_data, sampling_rate, nperseg = len(trace_data))
    
    return frequency, power_spectra



def window_band(frequencies: np.ndarray, spectrums: np.ndarray, f_min: float, f_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a subset of the frequency spectrum within a specified frequency band.

    Args:
        frequencies (np.ndarray): Array of frequency values.
        spectrums (np.ndarray): Array of spectral values corresponding to the frequencies.
        f_min (float): Minimum frequency of the band to extract (inclusive).
        f_max (float): Maximum frequency of the band to extract (inclusive).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - freq: Array of frequencies within the specified band.
            - spec: Array of spectral values corresponding to the extracted frequencies.
    """
    
    # Ensure input arrays are numpy arrays
    frequencies = np.asarray(frequencies)
    spectrums = np.asarray(spectrums)
    
    # Validate that frequencies and spectrums have the same length
    if frequencies.shape != spectrums.shape:
        raise ValueError("frequencies and spectrums must have the same shape.")

    # Extract indices for the specified frequency band
    indices = np.where((frequencies >= f_min) & (frequencies <= f_max))
    
    # Extract the frequencies and spectra within the specified band
    freq = frequencies[indices]
    spec = spectrums[indices]
    
    return  freq, spec



def calculate_moment_magnitude(
        wave_path: Path, 
        hypo_df: pd.DataFrame, 
        pick_df: pd.DataFrame, 
        station: pd.DataFrame, 
        calibration_path: Path, 
        event_id: str, 
        fig_path: Path, 
        fig_statement: bool = False
        ) -> Tuple[Dict[str, str], Dict[str,List]]:
    
    """
    Calculate the moment magnitude of an earthquake event and generate a spectral fitting profile.

    Args:
        wave_path (Path): Path to the directory containing waveform files.
        hypo_df (pd.DataFrame): DataFrame containing hypocenter information (latitude, longitude, depth).
        pick_df (pd.DataFrame): DataFrame containing pick information (arrival times).
        station (pd.DataFrame): DataFrame containing station information (latitude, longitude, elevation).
        calibration_path (Path): Path to the calibration files for instrument response removal.
        event_id (str): Unique identifier for the earthquake event.
        fig_path (Path): Path to save the generated figures.
        fig_statement (bool): Whether to generate and save figures (default is False).

    Returns:
        Tuple[Dict[str, str], Dict[str, List]]:
            - results (Dict[str, str]): A Dictionary containing calculated moment magnitude and related metrics.
            - fitting_result (Dict[str, List]): A dictionary Detailed fitting results for each station.
    """
    
    # Initialize figure if needed
    if fig_statement:
        try:
            num_stations = len(pick_df['Station'].unique())
            fig, axs= plt.subplots(num_stations*3, 2, figsize=(20,60))
            plt.subplots_adjust(hspace=0.5)
            axs[0,0].set_title("Phase Window", fontsize='20')
            axs[0,1].set_title("Spectra Fitting Profile", fontsize='20')
            counter = 0
        except Exception as e:
            logger.error(f"Error initializing figures for event {event_id}: {e}")
            fig_statement = False
    
    # Predefined parameters 
    R_PATTERN_P = 0.440 # P phase radiation pattern
    R_PATTERN_S = 0.600 # S phase radiation pattern
    k_P = 0.32 # kappa parameter for P phase
    k_S = 0.21 # kappa parameter for S phase
    LAYER_TOP   = [ [-2.0,0.0],[0.0, 1.2],[1.2, 6.1], [6.1, 14.1], [14.1,9999] ]
    VELOCITY_VP = [3.82, 4.50, 4.60, 6.20, 8.00]                                    # km/s
    VELOCITY_VS = [2.30, 2.53, 2.53, 3.44, 4.44]                                    # km/s
    DENSITY     = [ 2375.84,  2465.34, 2529.08, 2750.80, 2931.80]                   # kg/m3
    
    moments, corner_frequencies, source_radius = [], [], []
    fitting_result = {
        "ID":[],
        "Station":[],
        "F_corner_P":[],
        "F_corner_SV":[],
        "F_corner_SH":[],
        "Qfactor_P":[],
        "Qfactor_SV":[],
        "Qfactor_SH":[],
        "Omega_0_P(nms)":[],
        "Omega_0_SV(nms)":[],
        "Omega_0_SH(nms)":[],
        "RMS_e_P(nms)":[],
        "RMS_e_SV(nms)":[],
        "RMS_e_SH(nms)":[]
    }

    # Get hypocenter details
    origin_time = UTCDateTime(f"{hypo_df.Year.iloc[0]}-{int(hypo_df.Month.iloc[0]):02d}-{int(hypo_df.Day.iloc[0]):02d}T{int(hypo_df.Hour.iloc[0]):02d}:{int(hypo_df.Minute.iloc[0]):02d}:{float(hypo_df.T0.iloc[0]):012.9f}") 
    hypo_lat, hypo_lon , hypo_depth =  hypo_df.Lat.iloc[0], hypo_df.Lon.iloc[0], hypo_df.Depth.iloc[0]

    # Find the correct velocity and DENSITY value for the spesific layer depth
    for layer, (top, bottom) in enumerate(LAYER_TOP):
        if (top*1000)   <= hypo_depth <= (bottom*1000):
            velocity_P = VELOCITY_VP[layer]*1000  # velocity in m/s
            velocity_S = VELOCITY_VS[layer]*1000  # velocity in m/s
            DENSITY_value = DENSITY[layer]
    if not velocity_P:
        raise ValueError ("Hypo depth not within the defined layers.")

    # Read waveforms
    stream = read_waveforms(wave_path, event_id)
    st = stream.copy()
    
    # Start spectrum fitting and magnitude estimation
    for sta in list(pick_df.get("Station")):
        # Get the station coordinat
        sta_xyz = station[station.Stations == sta]
        sta_lat, sta_lon, sta_elev = sta_xyz.Lat.iloc[0], sta_xyz.Lon.iloc[0], sta_xyz.Elev.iloc[0]
        
        # Calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
        source_distance = np.sqrt(epicentral_distance**2 + (hypo_depth + sta_elev)**2)
        
        # Get the pick_df data for P arrival and S arrival
        pick_info = pick_df[pick_df.Station == sta].iloc[0]
        P_pick_time = UTCDateTime(
            f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
            f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_P):02d}:{float(pick_info.P_Arr_Sec):012.9f}"
        )
        S_pick_time = UTCDateTime(
            f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
            f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_S):02d}:{float(pick_info.S_Arr_Sec):012.9f}"
        )

        st2 = st.select(station = sta) # Select spesific seismograms from the stream

        if len(st2) < 3:
            logger.warning(f"Not all components available for station {sta} to calculate event {event_id} moment magnitude")
            continue
            
        st_removed = instrument_remove(st2, calibration_path, fig_path) # Remove instrument response
        hypo_ref = [hypo_lat, hypo_lon , -1*hypo_depth]  # depth must be in negative notation
        sta_ref = [sta_lat, sta_lon, sta_elev]
        take_off, total_tt, inc_angle = ref.calculate_inc_angle(hypo_ref, sta_ref, LAYER_TOP, VELOCITY_VP) # Calculate the incidence angle at station
        st_rotated = rotate_component(st_removed, azimuth, inc_angle) # do the component rotation from ZNE to LQT
        
        # Window the trace
        p_window_data, sv_window_data, sh_window_data, noise_window_data = window_trace(st_rotated, P_pick_time, S_pick_time)
        
        # Check the data quality (SNR must be above or equal to 1)
        if any(trace_snr(data, noise_window_data) <= 1 for data in [p_window_data, sv_window_data, sh_window_data]):
            logger.warning(f"SNR below threshold for station {sta} to calculate moment magnitude")
            continue
            
        # check sampling rate
        fs = 1 / st_rotated[0].stats.delta
        
        # calculate source spectra
        freq_P, spec_P = calculate_spectra(p_window_data, fs)
        freq_SV, spec_SV = calculate_spectra(sv_window_data, fs)
        freq_SH, spec_SH = calculate_spectra(sh_window_data, fs)
        
        # calculate the noise spectra
        freq_N, spec_N = calculate_spectra(noise_window_data, fs)
        
        # fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using grid search algorithm
        try:
            fit_P = fit.fit_spectrum_stochastic(freq_P, spec_P, abs(float(P_pick_time - origin_time)), F_MIN, F_MAX)
            fit_SV = fit.fit_spectrum_stochastic(freq_SV, spec_SV, abs(float(S_pick_time - origin_time)), F_MIN, F_MAX)
            fit_SH= fit.fit_spectrum_stochastic(freq_SH, spec_SH, abs(float(S_pick_time - origin_time)), F_MIN, F_MAX)
        except Exception as e:
            logger.error(f"Error during spectral fitting for event {event_id}: {e}")
            continue

        if None in [fit_SV, fit_SH, fit_P]:
            continue

        # fitting spectrum output
        Omega_0_P,  Q_factor_p,  f_c_P,  err_P,  x_fit_P,  y_fit_P  = fit_P
        Omega_0_SV, Q_factor_SV, f_c_SV, err_SV, x_fit_SV, y_fit_SV = fit_SV
        Omega_0_SH, Q_factor_SH, f_c_SH, err_SH, x_fit_SH, y_fit_SH = fit_SH
            
        # append the fitting spectrum output to the holder list
        Omega_0_P  = np.sqrt(Omega_0_P)
        Omega_0_SV = np.sqrt(Omega_0_SV)
        Omega_0_SH = np.sqrt(Omega_0_SH)
        
        err_P  = np.sqrt(err_P)
        err_SV = np.sqrt(err_SV)
        err_SH = np.sqrt(err_SH)

        # updating the fitting dict handler 
        fitting_result["ID"].append(event_id)
        fitting_result["Station"].append(sta)
        fitting_result["F_corner_P"].append(f_c_P)
        fitting_result["F_corner_SV"].append(f_c_SV)
        fitting_result["F_corner_SH"].append(f_c_SH)
        fitting_result["Qfactor_P"].append(Q_factor_p)
        fitting_result["Qfactor_SV"].append(Q_factor_SV)
        fitting_result["Qfactor_SH"].append(Q_factor_SH)
        fitting_result["Omega_0_P(nms)"].append((Omega_0_P*1e9))
        fitting_result["Omega_0_SV(nms)"].append((Omega_0_SV*1e9))
        fitting_result["Omega_0_SH(nms)"].append((Omega_0_SH*1e9))
        fitting_result["RMS_e_P(nms)"].append((err_P*1e9))
        fitting_result["RMS_e_SV(nms)"].append((err_SV*1e9))
        fitting_result["RMS_e_SH(nms)"].append((err_SH*1e9))

        # create figure
        if fig_statement:
            # frequency window for plotting purposes
            f_min_plot = 0.75
            f_max_plot = 100
            freq_P, spec_P = window_band(freq_P, spec_P, f_min_plot, f_max_plot)
            freq_SV, spec_SV = window_band(freq_SV, spec_SV, f_min_plot, f_max_plot)
            freq_SH, spec_SH = window_band(freq_SH, spec_SH, f_min_plot, f_max_plot)
            freq_N, spec_N = window_band(freq_N, spec_N, f_min_plot, f_max_plot)

            # dinamic window parameter
            s_p_time = float(S_pick_time - P_pick_time)    
            time_after_pick_p = 0.80 * s_p_time
            time_after_pick_s = 1.75 * s_p_time
            
            try:
                # plot for phase windowing
                # 1. For P phase or vertical component
                tr_L = st_rotated.select(component = 'L')[0]
                start_time = tr_L.stats.starttime
                before = (P_pick_time - start_time) - 2.0
                after = (S_pick_time - start_time) + 6.0
                tr_L.trim(start_time+before, start_time+after)
                axs[counter][0].plot(tr_L.times(), tr_L.data, 'k')
                axs[counter][0].axvline( x= (P_pick_time - tr_L.stats.starttime ), color='r', linestyle='-', label='P arrival')
                axs[counter][0].axvline( x= (S_pick_time - tr_L.stats.starttime ), color='b', linestyle='-', label='S arrival')
                axs[counter][0].axvline( x= (P_pick_time - TIME_BEFORE_PICK -  tr_L.stats.starttime), color='g', linestyle='--')
                axs[counter][0].axvline( x= (P_pick_time + time_after_pick_p - tr_L.stats.starttime), color='g', linestyle='--', label='P phase window')
                axs[counter][0].axvline( x= (P_pick_time - NOISE_DURATION -  tr_L.stats.starttime), color='gray', linestyle='--')
                axs[counter][0].axvline( x= (P_pick_time - NOISE_PADDING  - tr_L.stats.starttime), color='gray', linestyle='--', label='Noise window')
                axs[counter][0].set_title("{}_BH{}".format(tr_L.stats.station, tr_L.stats.component), loc="right",va='center')
                axs[counter][0].legend()
                axs[counter][0].set_xlabel("Relative Time (s)")
                axs[counter][0].set_ylabel("Amp (m)")
               
                # 2. For SV phase or radial component
                axis = counter + 1
                tr_Q = st_rotated.select(component = 'Q')[0]
                start_time = tr_Q.stats.starttime
                before = (P_pick_time - start_time) - 2.0
                after = (S_pick_time - start_time) + 6.0
                tr_Q.trim(start_time+before, start_time+after)
                axs[counter+1][0].plot(tr_Q.times(), tr_Q.data, 'k')
                axs[counter+1][0].axvline( x= (P_pick_time - tr_Q.stats.starttime ), color='r', linestyle='-', label='P arrival')
                axs[counter+1][0].axvline( x= (S_pick_time - tr_Q.stats.starttime), color='b', linestyle='-', label='S arrival')
                axs[counter+1][0].axvline( x= (S_pick_time - TIME_BEFORE_PICK -  tr_Q.stats.starttime  ), color='g', linestyle='--')
                axs[counter+1][0].axvline( x= (S_pick_time + time_after_pick_s - tr_Q.stats.starttime ), color='g', linestyle='--', label='SV phase window')
                axs[counter+1][0].axvline( x= (P_pick_time - NOISE_DURATION -  tr_Q.stats.starttime), color='gray', linestyle='--')
                axs[counter+1][0].axvline( x= (P_pick_time - NOISE_PADDING  - tr_Q.stats.starttime), color='gray', linestyle='--', label='Noise window')
                axs[counter+1][0].set_title("{}_BH{}".format(tr_Q.stats.station, tr_Q.stats.component), loc="right",va='center')
                axs[counter+1][0].legend()
                axs[counter+1][0].set_xlabel("Relative Time (s)")
                axs[counter+1][0].set_ylabel("Amp (m)")
                
                # 3. For SH phase or transverse component
                
                tr_T = st_rotated.select(component = 'T')[0]
                start_time = tr_T.stats.starttime
                before = (P_pick_time - start_time) - 2.0
                after = (S_pick_time - start_time) + 6.0
                tr_T.trim(start_time+before, start_time+after)
                axs[counter+2][0].plot(tr_T.times(), tr_T.data, 'k')
                axs[counter+2][0].axvline( x= (P_pick_time - tr_T.stats.starttime ), color='r', linestyle='-', label='P arrival')
                axs[counter+2][0].axvline( x= (S_pick_time - tr_T.stats.starttime), color='b', linestyle='-', label='S arrival')
                axs[counter+2][0].axvline( x= (S_pick_time - TIME_BEFORE_PICK -  tr_T.stats.starttime  ), color='g', linestyle='--')
                axs[counter+2][0].axvline( x= (S_pick_time + time_after_pick_s - tr_T.stats.starttime ), color='g', linestyle='--', label='SH phase window')
                axs[counter+2][0].axvline( x= (P_pick_time - NOISE_DURATION -  tr_T.stats.starttime), color='gray', linestyle='--')
                axs[counter+2][0].axvline( x= (P_pick_time - NOISE_PADDING  - tr_T.stats.starttime), color='gray', linestyle='--', label='Noise window')
                axs[counter+2][0].set_title("{}_BH{}".format(tr_T.stats.station, tr_T.stats.component), loc="right",va='center')
                axs[counter+2][0].legend()
                axs[counter+2][0].set_xlabel("Relative Time (s)")
                axs[counter+2][0].set_ylabel("Amp (m)")
               
                # plot the spectra (P, SV, SH and Noise spectra)
                # 1. For P spectra
                axs[counter][1].loglog(freq_P, spec_P, color='black', label='P spectra')
                axs[counter][1].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
                axs[counter][1].loglog(x_fit_P, y_fit_P, 'b-', label='Fitted P Spectra')
                axs[counter][1].set_title("{}_BH{}".format(tr_L.stats.station, tr_L.stats.component), loc="right",va='center')
                axs[counter][1].legend()
                axs[counter][1].set_xlabel("Frequencies (Hz)")
                axs[counter][1].set_ylabel("Amp (m/Hz)")
               
               
                # 2. For SV spectra
                axs[counter+1][1].loglog(freq_SV, spec_SV, color='black', label='SV spectra')
                axs[counter+1][1].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
                axs[counter+1][1].loglog(x_fit_SV, y_fit_SV, 'b-', label='Fitted SV Spectra')
                axs[counter+1][1].set_title("{}_BH{}".format(tr_Q.stats.station, tr_Q.stats.component), loc="right",va='center')
                axs[counter+1][1].legend()
                axs[counter+1][1].set_xlabel("Frequencies (Hz)")
                axs[counter+1][1].set_ylabel("Amp (m/Hz)")
                
                
                # 3. For SH spectra
                axs[counter+2][1].loglog(freq_SH, spec_SH, color='black', label='SH spectra')
                axs[counter+2][1].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
                axs[counter+2][1].loglog(x_fit_SH, y_fit_SH, 'b-', label='Fitted SH Spectra')
                axs[counter+2][1].set_title("{}_BH{}".format(tr_T.stats.station, tr_T.stats.component), loc="right",va='center')
                axs[counter+2][1].legend()
                axs[counter+2][1].set_xlabel("Frequencies (Hz)")
                axs[counter+2][1].set_ylabel("Amp (m/Hz)")

                counter +=3
            except Exception as e:
                logger.error(f"Failed to plot the fitting spectral for event {event_id} : {e}") 

        # calculate the moment magnitude
        try:
            # calculate the  resultant omega
            omega_P = Omega_0_P
            omega_S = (Omega_0_SV**2 + Omega_0_SH**2)**0.5
         
            ## calculate seismic moment
            M_0_P = 4.0 * np.pi * DENSITY_value * (velocity_P ** 3) * source_distance * \
                    omega_P / \
                    (R_PATTERN_P)                                                   # should it be multipled by 2 ??
                    
            M_0_S = 4.0 * np.pi * DENSITY_value * (velocity_S ** 3) * source_distance * \
                    omega_S / \
                    (R_PATTERN_S)                                                   # should it be multipled by 2 ??
            
            # calculate source radius
            r_P = k_P * velocity_P * f_c_P # result in meter, times 3 because it is a three components
            r_S = 2 * k_S * velocity_S /(f_c_SV + f_c_SH) # result in meter, times 3 because it is a three components
            
            # extend the moments object holder to calculate the moment magnitude
            moments.extend([M_0_P, M_0_S]) 
            
            # calculate corner frequency mean
            corner_freq_S = (f_c_SV + f_c_SH)/2
            corner_frequencies.extend([f_c_P, corner_freq_S])
            
            # extend the source radius
            source_radius.extend([r_P, r_S])        
     
        except Exception as e:
            logger.error(f"Failed to calculate seismic moment for event {event_id} : {e}")
            continue
            
    # Calculate the seismic moment viabnv  basic statistics.
    moments = np.array(moments)
    moment = moments.mean()
    moment_std = moments.std()
    
    ## calculate the corner frequencies via basic statistics.
    corner_frequencies = np.array(corner_frequencies)
    corner_frequency = corner_frequencies.mean()
    corner_frequency_std = corner_frequencies.std()

    # Calculate the source radius.
    source_radius = np.array(source_radius)
    source_rad = source_radius.mean()
    source_radius_std = source_radius.std()
    
    # Calculate the stress drop of the event based on the average moment and source radius
    stress_drop = ((7 * moment) / (16 * (source_rad * 0.001) ** 3))*1e-14
    stress_drop_std = np.sqrt((stress_drop ** 2) * (((moment_std ** 2) / (moment ** 2)) + \
    (9 * source_rad * source_radius_std ** 2)))   
        
    # Calculate the final moment magnitude
    Mw  = ((2.0 / 3.0) * np.log10(moment)) - 6.07
    
    # calculate moment magnitude from seismic moment standard deviation 
    Mw_std = 2.0 / 3.0 * moment_std / (moment * np.log(10))
 
    results = {"ID":[f"{event_id}"], 
                "Fc":[f"{corner_frequency}"],
                "Fc_std":[f"{corner_frequency_std}"],
                "Mw":[f"{Mw}"],
                "Mw_std":[f"{Mw_std}"],
                "Src_rad(m)":[f"{source_rad}"],
                "Src_rad_std":[f"{source_radius_std}"],
                "Stress_drop(bar)":[f"{stress_drop}"],
                "Stress_drop_std":[f"{stress_drop_std}"]
                }
                
    if fig_statement : 
        fig.suptitle(f"Event {event_id} Spesctral Fitting Profile", fontsize='24', fontweight='bold')
        #plt.title("Event {} Spectral Fitting Profile".format(ID), fontsize='20')
        plt.savefig(fig_path.joinpath(f"event_{event_id}.png"))
    
    return results, fitting_result



def main():
    # initialize input and output path
    wave_path       = Path(r"G:\SEML\DATA TRIMMING\EVENT DATA TRIM\ALL COMBINED")                           # trimmed waveform location
    hypo_input      = Path(r"G:\SEML\CATALOG HYPOCENTER\catalog\hypo_reloc.xlsx")                           # relocated catalog
    sta_input       = Path(r"G:\SEML\STATION AND VELOCITY DATA\SEML_station.xlsx")                          # station file
    pick_input      = Path(r"G:\SEML\CATALOG HYPOCENTER\catalog\catalog_picking.xlsx")                      # catalog picking
    calibration     = Path(r"G:\SEML\SEISMOMETER INSTRUMENT CORRECTION\CALIBRATION")                        # calibration file
    mw_result       = Path(r"G:\SEML\MAGNITUDE CALCULATION\MW")                                             # mw result location
    fig_output      = Path(r"G:\SEML\MAGNITUDE CALCULATION\MW\fig_out")                                     # saved figure location
    
    # Call the function to start calculating moment magnitude
    mw_result_df, mw_fitting_df, output_name = start_calculate(wave_path, hypo_input, sta_input, pick_input, calibration, fig_output)

    # save and set dataframe index
    mw_result_df.to_excel(mw_result.joinpath(f"{output_name}_result.xlsx"), index = False)
    mw_fitting_df.to_excel(mw_result.joinpath(f"{output_name}_fitting_result.xlsx"), index = False)
    
    return None



if __name__ == "__main__" :
    main()