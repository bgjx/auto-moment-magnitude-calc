import numpy as np
from skopt import gp_minimize
from skopt.space import Real , Integer
from skopt.utils import use_named_args
from scipy.stats import uniform

# function for spectrum windowing within spesific f_band
def window_band(frequencies, spectrums, f_min, f_max):
    indices = np.where((frequencies >= f_min) & (frequencies <= f_max))
    freq = frequencies[indices]
    spec = spectrums[indices]
    return  freq, spec
    
# function for calculating the source spectrum (spectrum model) 
def calculate_source_spectrum(frequencies, omega_0, Q, corner_frequency, 
    traveltime):
    """
    After Abercrombie (1995) and Boatwright (1980).
    Abercrombie, R. E. (1995). Earthquake locations using single-station deep
    borehole recordings: Implications for microseismicity on the San Andreas
    fault in southern California. Journal of Geophysical Research, 100,
    24003–24013.
    Boatwright, J. (1980). A spectral theory for circular seismic sources,
    simple estimates of source dimension, dynamic stress drop, and radiated
    energy. Bulletin of the Seismological Society of America, 70(1).
    The used formula is:
        Omega(f) = (Omege(0) * e^(-pi * f * T / Q)) / (1 + (f/f_c)^4) ^ 0.5
    :param frequencies: Input array to perform the calculation on.
    :param omega_0: Low frequency amplitude in [meter x second].
    :param corner_frequency: Corner frequency in [Hz].
    :param Q: Quality factor.
    :param traveltime: Traveltime in [s].
    """
    num = omega_0 * np.exp(-np.pi * frequencies * traveltime / Q)
    denom = (1 + (frequencies / corner_frequency) ** 4)**0.5
    return num / denom


# fitting spectrum using Grid search algorithm
def fit_spectrum_systematic (frequencies, spectrums, traveltime, f_min, f_max):
    # windowing frequencies and spectrum within f band    
    freq, spectrum = window_band(frequencies, spectrums, f_min, f_max)
    
    # setting initial guess
    peak_omega = spectrum.max()
    omega_0 = np.linspace(peak_omega/10, peak_omega*10, 100)
    Q_factor = np.linspace(50, 2500, 50)
    f_c = np.linspace(0.75, 30, 50)
    
    # rms and error handler
    error = np.inf
    
    # define callable function
    def f(freqs, omega, qfactor, f_cor):
        return calculate_source_spectrum(freqs, omega, qfactor, f_cor, traveltime)
        
    # start guessing
    for i in range(len(omega_0)):
        for j in range(len(Q_factor)):
            for k in range(len(f_c)):
                fwd = f(freq, omega_0[i], Q_factor[j], f_c[k])
                rms_e = np.sqrt(np.mean((fwd - spectrum)**2))
                if rms_e < error:
                    error = rms_e
                    omega_0_fit = omega_0[i]
                    Q_factor_fit = Q_factor[j]
                    f_c_fit = f_c[k]
                    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = f(x_tuned, omega_0_fit, Q_factor_fit, f_c_fit) 
                    
    return omega_0_fit, Q_factor_fit, f_c_fit, rms_e, x_tuned, y_tuned

# fitting spectrum using Random search algorithm
def fit_spectrum_stochastic (frequencies, spectrums, traveltime, f_min, f_max):
    # windowing frequencies and spectrum within f band    
    freq, spectrum = window_band(frequencies, spectrums, f_min, f_max)
    
    # setting initial guess
    peak_omega = spectrum.max()
    omega_0_range = (peak_omega/10, peak_omega*10)
    Q_factor_range = (50, 3000)
    f_c_range = (0.75, 30)
    
    # rms and error handler
    best_params = {'omega_0': None, 'Q_factor': None, 'f_c': None}
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
            best_params['omega_0'] = omega_0
            best_params['Q_factor'] = Q_factor
            best_params['f_c'] = f_c
                    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = f(x_tuned, best_params['omega_0'], best_params['Q_factor'], best_params['f_c']) 
                    
    return best_params['omega_0'], best_params['Q_factor'], best_params['f_c'], best_rms_e, x_tuned, y_tuned


# fitting spectrum with bayesian optimization
def fit_spectrum_bayes_opt (frequencies, spectrums, traveltime, f_min, f_max):
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
    omega_0_best, Q_factor_best, f_c_best = result.x
    best_error = result.fun
    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = calculate_source_spectrum(x_tuned, omega_0_best, Q_factor_best, f_c_best, traveltime) 
                    
    return omega_0_best, Q_factor_best, f_c_best, best_error, x_tuned, y_tuned
    
# # fitting spectrum function using levenberg-marquardt algorithm
# def fit_spectrum_first(spectrum, frequencies, traveltime, initial_omega_0,
    # initial_f_c):
    # """
    # Fit a theoretical source spectrum to a measured source spectrum.
    # Uses a Levenburg-Marquardt algorithm.
    # :param spectrum: The measured source spectrum.
    # :param frequencies: The corresponding frequencies.
    # :para traveltime: Event traveltime in [s].
    # :param initial_omega_0: Initial guess for Omega_0.
    # :param initial_f_c: Initial guess for the corner frequency.
    # :param initial_q: initial quality factor
    # :returns: Best fits and standard deviations.
        # (Omega_0, f_c, Omega_0_std, f_c_std)
        # Returns None, if the fit failed.
    # """
    # def f(frequencies, omega_0, f_c):
        # return calculate_source_spectrum(frequencies, omega_0, f_c,
                # Qfactor, traveltime)
    # popt, pcov = scipy.optimize.curve_fit(f, frequencies, spectrum, \
        # p0=list([initial_omega_0, initial_f_c]), maxfev=100000)        # maxfev is the maximum number of function calls allowed during the optimization
    # # p0 is the initial guest that will be optimized by the fit method
    # # popt is the optimezed parameters and the pcov is the covariance matrix
    
    # x_fit=frequencies
    # y_fit= f(x_fit, *popt)
    
    # if popt is None:
        # return None
    # return popt[0], popt[1], pcov[0, 0], pcov[1, 1], x_fit,y_fit


if __name__ == "__main__":
    print("test")