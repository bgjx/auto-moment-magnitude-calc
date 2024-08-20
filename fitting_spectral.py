from typing import Dict, Tuple, List, Optional
import numpy as np
from skopt import gp_minimize
from skopt.space import Real , Integer
from skopt.utils import use_named_args
from scipy.stats import uniform


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
    indices = np.where((frequencies >= f_min) & (frequencies <= f_max))
    freq = frequencies[indices]
    spec = spectrums[indices]
    return  freq, spec
    
    
    
def calculate_source_spectrum(frequencies: np.ndarray,
                                omega_0: float,
                                Q: float,
                                corner_frequency: float,
                                traveltime: float
                                ) -> np.ndarray:
    """
    Calculate the source spectrum based on Abercrombie (1995) and Boatwright (1980).
    Abercrombie, R. E. (1995). Earthquake locations using single-station deep
    borehole recordings: Implications for microseismicity on the San Andreas
    fault in southern California. Journal of Geophysical Research, 100,
    24003–24013.
    Boatwright, J. (1980). A spectral theory for circular seismic sources,
    simple estimates of source dimension, dynamic stress drop, and radiated
    energy. Bulletin of the Seismological Society of America, 70(1).
    
    Args:
        frequencies (np.ndarray): Array of frequency values.
        omega_0 (float): Value of the omega zero (spectral flat level).
        Q (float): Quality factor for amplitude attenuation.
        corner_frequency (float): The corner frequency.
        traveltime(float): Value of the phase travel time.
        
    Returns:
        np.ndarray: An array of theoretical calculated source spectrum.
    """
    num = omega_0 * np.exp(-np.pi * frequencies * traveltime / Q)
    denom = (1 + (frequencies / corner_frequency) ** 4)**0.5
    return num / denom
    
    
    
def fit_spectrum_systematic (frequencies: np.ndarray,
                                spectrums: np.ndarray,
                                traveltime: float,
                                f_min: float,
                                f_max: float
                                ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    A systematic parameter optimization process using grid search to obtain the best-fitting model for the observed spectrum.
    
    Args:
        frequencies (np.ndarray): Array of frequency values.
        omega_0 (float): Value of the omega zero (spectral flat level).
        Q (float): Quality factor for amplitude attenuation.
        corner_frequency (float): The corner frequency.
        traveltime(float): Value of the phase travel time.
        f_min (float): Minimum frequency of the band to be calculated.
        f_max (float): Maximum frequency of the band to be calculated.
        
    Returns:
        Tuple[float, float, float, float, np.ndarray, np.ndarray]:
            - omega_0_fit: The optimized value of omega 0 (spectral flat level) obtained from fitting the model.
            - q_factor_fit: The optimized value of quality factor obtained from fitting the model.
            - f_c_fit: The optimized value of frequency corner obtained from fitting the model.
            - best_rms_e : The best rms value obtained from model fitting process.
            - x_tuned : Array of frequencies in spesific resolution and frequencies band.
            - y_tuned: Array of spectrum computed from the model equation using optimized variable.
    """
    # windowing frequencies and spectrum within f band    
    freq, spectrum = window_band(frequencies, spectrums, f_min, f_max)
    
    # setting initial guess
    peak_omega = spectrum.max()
    omega_0 = np.linspace(peak_omega/10, peak_omega*10, 100)
    Q_factor = np.linspace(50, 2500, 50)
    f_c = np.linspace(0.75, 30, 50)
    
    # rms and error handler
    best_rms_e = np.inf
    
    # define callable function
    def f(freqs, omega, qfactor, f_cor):
        return calculate_source_spectrum(freqs, omega, qfactor, f_cor, traveltime)
        
    # start guessing
    for i in range(len(omega_0)):
        for j in range(len(Q_factor)):
            for k in range(len(f_c)):
                fwd = f(freq, omega_0[i], Q_factor[j], f_c[k])
                rms_e = np.sqrt(np.mean((fwd - spectrum)**2))
                if rms_e < best_rms_e:
                    best_rms_e = rms_e
                    omega_0_fit = omega_0[i]
                    Q_factor_fit = Q_factor[j]
                    f_c_fit = f_c[k]
                    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = f(x_tuned, omega_0_fit, Q_factor_fit, f_c_fit) 
                    
    return omega_0_fit, Q_factor_fit, f_c_fit, best_rms_e, x_tuned, y_tuned
    
    
    
def fit_spectrum_stochastic (frequencies: np.ndarray,
                                spectrums: np.ndarray,
                                traveltime: float,
                                f_min: float,
                                f_max: float
                                ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    A stochastic parameter optimization process using random search to obtain the best-fitting model for the observed spectrum.
    
    Args:
        frequencies (np.ndarray): Array of frequency values.
        omega_0 (float): Value of the omega zero (spectral flat level).
        Q (float): Quality factor for amplitude attenuation.
        corner_frequency (float): The corner frequency.
        traveltime(float): Value of the phase travel time.
        f_min (float): Minimum frequency of the band to be calculated.
        f_max (float): Maximum frequency of the band to be calculated.
        
    Returns:
        Tuple[float, float, float, float, np.ndarray, np.ndarray]:
            - omega_0_fit: The optimized value of omega 0 (spectral flat level) obtained from fitting the model.
            - q_factor_fit: The optimized value of quality factor obtained from fitting the model.
            - f_c_fit: The optimized value of frequency corner obtained from fitting the model.
            - best_rms_e : The best rms value obtained from model fitting process.
            - x_tuned : Array of frequencies in spesific resolution and frequencies band.
            - y_tuned: Array of spectrum computed from the model equation using optimized variable.
    """
    # windowing frequencies and spectrum within f band    
    freq, spectrum = window_band(frequencies, spectrums, f_min, f_max)
    
    # setting initial guess
    peak_omega = spectrum.max()
    omega_0_range = (peak_omega/10, peak_omega*10)
    Q_factor_range = (50, 3000)
    f_c_range = (0.75, 30)
    
    # rms and error handler
    omega_0_fit, Q_factor_fit, f_c_fit = np.inf, np.inf, np.inf
    best_rms_e = np.inf
    
    # define callable function
    def f(freqs, omega, qfactor, f_cor):
        return calculate_source_spectrum(freqs, omega, qfactor, f_cor, traveltime)
        
    # start guessing
    # set number of iteration
    n_iter = 5000
    for _ in range(n_iter):
        omega_0 = uniform.rvs(*omega_0_range)
        Q_factor = uniform.rvs(*Q_factor_range)
        f_c = uniform.rvs(*f_c_range)
        
        fwd = f(freq, omega_0, Q_factor, f_c)
        rms_e = np.sqrt(np.mean((fwd - spectrum)**2))
        
        if rms_e < best_rms_e:
            best_rms_e = rms_e
            omega_0_fit = omega_0
            Q_factor_fit = Q_factor
            f_c_fit = f_c
                    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = f(x_tuned, omega_0_fit, Q_factor_fit, f_c_fit) 
                    
    return omega_0_fit, Q_factor_fit, f_c_fit, best_rms_e, x_tuned, y_tuned
    
    
    
def fit_spectrum_bayes_opt (frequencies: np.ndarray,
                                spectrums: np.ndarray,
                                traveltime: float,
                                f_min: float,
                                f_max: float
                                ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    A Probabilistic parameter optimization process using bayesian optimization to obtain the best-fitting model for the observed spectrum.
    
    Args:
        frequencies (np.ndarray): Array of frequency values.
        omega_0 (float): Value of the omega zero (spectral flat level).
        Q (float): Quality factor for amplitude attenuation.
        corner_frequency (float): The corner frequency.
        traveltime(float): Value of the phase travel time.
        f_min (float): Minimum frequency of the band to be calculated.
        f_max (float): Maximum frequency of the band to be calculated.
        
    Returns:
        Tuple[float, float, float, float, np.ndarray, np.ndarray]:
            - omega_0_fit: The optimized value of omega 0 (spectral flat level) obtained from fitting the model.
            - q_factor_fit: The optimized value of quality factor obtained from fitting the model.
            - f_c_fit: The optimized value of frequency corner obtained from fitting the model.
            - best_rms_e : The best rms value obtained from model fitting process.
            - x_tuned : Array of frequencies in spesific resolution and frequencies band.
            - y_tuned: Array of spectrum computed from the model equation using optimized variable.
    """
    # windowing frequencies and spectrum within f band    
    freq, spectrum = window_band(frequencies, spectrums, f_min, f_max)

    # define the search space for hyperparameters
    peak_omega = spectrum.max()
    space = [
        Real(peak_omega/10, peak_omega*10, name = 'omega_0'),
        Real(50, 2500, name = 'Q_factor'),
        Real(0.75, 30, name = 'f_c' )
    ]
    
    # define the objective_function
    @use_named_args(space)
    def objective(omega_0, Q_factor, f_c):
        def f(freqs, omega, qfactor, f_cor):
            return calculate_source_spectrum(freqs, omega, qfactor, f_cor, traveltime)
    
        fwd = f(freq, omega_0, Q_factor, f_c)
        rms_e = np.sqrt(np.mean((fwd - spectrum)**2))
        
        return rms_e
    
    # perform the Bayesian optimization
    result = gp_minimize(objective, space, n_calls = 15, random_state = 42)
    
    # extract the best parameters and result
    omega_0_fit, Q_factor_fit, f_c_fit = result.x
    best_rms_e = result.fun
    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = calculate_source_spectrum(x_tuned, omega_0_fit, Q_factor_fit, f_c_fit, traveltime) 
                    
    return omega_0_fit, Q_factor_fit, f_c_fit, best_rms_e, x_tuned, y_tuned


if __name__ == "__main__":
    print("test")