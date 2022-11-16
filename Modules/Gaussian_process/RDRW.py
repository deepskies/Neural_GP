__author__ = "egorssed"

#Basic imports
import numpy as np
from functools import partial
import scipy

#JAX
import jax
import jax.numpy as jnp
# Probably should specify in the outer script, not the inner modules
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

# for Matern kernel with Modified Bessel of the 2nd kind (tfp>=0.17.0-dev20220322)
import tensorflow_probability as tfp


#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(max_thread_numbers)

# This notebook is meant to handle generation of Reverberating Damped Random Walk Gaussian-Markov process

#softening
eps = 1e-8


def Matern_spectrum(frequency, Variance, Correlation_length, nu):
    """
    .. math::
        S(f)=\\sigma^2\\frac{2\\pi \\Gamma(nu+0.5)}{\\nu \\Gamma(nu)}
        \\frac{\\tau}{(1+4\\pi^2f^2\\tau^2/2\\nu)^{-(\\nu+0.5)}}

    Power spectrum of a 1d process with Matern covariance
    Parameters
    ----------
    frequency: float [1/days]
    Variance: float
        Variance of a sampled point
    Correlation_length: float [days]
        Measure of spatial correlation between separated points
    nu: float
        Order of the covariance (order of modified Bessel in configuration space or power slope in Fourier)

    Returns
    -------
    Power: float
        Power of the process on a given 'frequency'

    """
    Normalisation = jnp.sqrt(2 * np.pi / nu) * jnp.exp(
        jax.scipy.special.gammaln(nu + 0.5) - jax.scipy.special.gammaln(nu))
    Power_decay = Correlation_length * jnp.power(1 + (2 * np.pi * frequency * Correlation_length) ** 2 / (2 * nu),
                                                 -(nu + 0.5))


    return Variance * Normalisation * Power_decay


@partial(jax.jit, static_argnums=(3,))
def Matern_kernel(time_difference, Variance, Correlation_length, nu):
    """
    .. math::
        C_{\\nu}(d)=\\sigma^2 \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}
        \\left(\\sqrt{2\\nu}\\frac{d}{\\tau}\\right)^\\nu K_{\\nu}(\\sqrt{2\\nu}\\frac{d}{\\tau})

    Autocorrelation function of a 1d process with Matern covariance.
    The function uses analytic expressions for nu in [0.5,1.5,2.5]
    and expression via Modified Bessel Function of the 2nd kind in other cases
    Parameters
    ----------
    time_difference: float [days]
        Difference in time between two points, for which correlation is estimated
    Variance: float
        Variance of a sampled point
    Correlation_length: float [days]
        Measure of spatial correlation between separated points
    nu: float

        Order of the covariance (order of modified Bessel in configuration space or power slope in Fourier)
    -------
    Returns
    Correlation: float
        Correlation between points t_i and t_j
    """

    # Matern covariance is isotropic
    time_difference = jnp.abs(time_difference)

    # nu that result in analytic Bessel function
    if nu == 0.5:
        return Variance * jnp.exp(-time_difference / Correlation_length)
    elif nu == 1.5:
        arg = np.sqrt(3) * time_difference / Correlation_length
        return Variance * (1 + arg) * jnp.exp(-arg)
    elif nu == 2.5:
        arg = np.sqrt(5) * time_difference / Correlation_length
        return Variance * (1 + arg + arg ** 2 / 3) * jnp.exp(-arg)

    # non-analytic forms
    Gamma_func = np.exp(scipy.special.gammaln(nu))
    Normalisation_factor = np.power(2, 1 - nu) / Gamma_func
    arg = np.sqrt(2 * nu) * time_difference / Correlation_length
    get_general_Matern = lambda arg: Normalisation_factor * jnp.power(arg, nu) * \
                                     tfp.substrates.jax.math.bessel_kve(v=nu, z=arg) * jnp.exp(-arg)

    # we have Norm*0*inf=1 for time_difference==0, so we need to use conditional operator here
    Autocorrelation = jax.lax.cond(time_difference == 0, lambda arg: 1., get_general_Matern, arg)

    return Variance * Autocorrelation

def RDRW_kernel(time_difference, Variance, Correlation_time, Reverberation_time):
    """
    Autocorrelation function of Reverberating Damped Random Walk process. This is a difference of two Mater[nu=0.5] kernels.
    Formally it is subtraction of two laplacian kernels

    .. math::
        R_{FF}(\\Delta t) = \\sigma^2  \\frac{1}{t_{corr}- t_{rev}}
            \\left(t_{corr} e^{-\\frac{|\\Delta t|}{t_{corr}}} - t_{rev} e^{-\\frac{|\\Delta t|}{t_{rev}}}\\right),

    Parameters
    ----------
    time_difference: float [days]
        Difference in time between two points, for which correlation is estimated
    Variance: float
        Variance of a sampled point
    Correlation_time: float [days]
        Measure of global correlation between separated points. Defines global evolution.
    Reverberation_time: float [days]
        Measure of local correlation between separated points. Defines locall evolution.
        It is assumed that Reverberation_time<Correlation_time

    Returns
    -------
    R_{FF}(t_i - t_j) : float
        Correlation of radiation on time stamps t_i and t_j (stationary,isotropic)
    """


    Normalisation = 1. / (Correlation_time  - Reverberation_time)
    Time_dependence = Correlation_time * jnp.exp(-jnp.abs(time_difference) / (Correlation_time + eps)) - \
                      Reverberation_time * jnp.exp(-jnp.abs(time_difference) / (Reverberation_time + eps))


    return Variance*Normalisation*Time_dependence



@partial(jax.jit, static_argnums=(4))
def Cholesky_of_Covariance_matrix(time_array: jnp.ndarray, Normalised_errors: jnp.ndarray, Correlation_time: float,
                                  Reverberation_time: float, Curve_type='RDRW'):
    """
    Cholesky decomposition of Covariance matrix with unit variance

    Parameters
    ----------
    time_array: array (N,) [days]
        time stamps, on which the process should be sampled
    Normalised_errors: array (N,)
        Observation errors normalised by the Gaussian process' variance
    Correlation_time: float [days]
        Measure of global correlation between separated points. Defines global evolution.
    Reverberation_time: float [days]
        Measure of local correlation between separated points. Defines local evolution.
        It is assumed that Reverberation_time<Correlation_time
    Curve_type: 'RDRW' or 'DRW'
        Assumes either Reverberating Damped Random Walk (RDRW) kernel or DRW kernel (Matern[0.5])

    Returns
    -------
        L: ndarray (N,N)
            Low triangular decomposition of the Covariance matrix
    """

    N = len(time_array)

    # Distance between time points for autocorrelation function
    x, y = jnp.meshgrid(time_array, time_array)
    time_difference = x - y

    if Curve_type == 'RDRW':
        get_Covariance = lambda time_difference: RDRW_kernel(time_difference, 1.,
                                                             Correlation_time, Reverberation_time)
    elif Curve_type == 'DRW':
        # DRW is sampled from Matern Covariance with order 0.5, where
        get_Covariance = lambda time_difference: Matern_kernel(time_difference, 1., Correlation_time, 0.5)
    else:
        raise ValueError('Wrong light curve type. Pick one of "RDRW" or "DRW"')

    # Covariance kernel (unit variance in a time point)
    Covariance_matrix = get_Covariance(time_difference) + jnp.diag(Normalised_errors) + np.eye(N) * eps
    # Now use Cholesky decomposition, but might change to sparce matrices jax.scipy.sparse.linalg
    Low_triangular_Covariance = jax.scipy.linalg.cholesky(Covariance_matrix, lower=True, overwrite_a=True)

    return Low_triangular_Covariance

def Mahalanobis_distance(values_vector, Mean, Low_triangular_Covariance):

    full_one = jnp.ones_like(values_vector)

    # Squared Mah. distance
    Norm = (values_vector - Mean * full_one).T @ (
        jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), values_vector - Mean * full_one))

    return jnp.sqrt(Norm)

def get_GP_sampler(time_array, Correlation_time: float, Reverberation_time: float, Curve_type='RDRW'):
    """
    This function returns a function F[t_corr,t_rev](Mean,Variance,Random_seed)
    precompiled for Correlation_time t_corr and Reverberation_time t_rev.
    This function can sampled a Gaussian process for any Mean,Variance and Random_seed in no time

    Parameters
    ----------
    time_array: array [days]
        time stamps, on which the process should be sampled
    Correlation_time: float [days]
        Measure of global correlation between separated points. Defines global evolution.
    Reverberation_time: float [days]
        Measure of local correlation between separated points. Defines local evolution.
        It is assumed that Reverberation_time<Correlation_time
    Curve_type: 'RDRW' or 'DRW'
        Assumes either Reverberating Damped Random Walk (RDRW) kernel or DRW kernel (Matern[0.5])

    Returns
    -------
    F[t_corr,t_rev]: function
        F[t_corr,t_rev](Mean,Variance,Random_seed) precompiled for Correlation_time t_corr and Reverberation_time t_rev.
    """

    # Use Cholesky decomposition to get lower triangle matrix describing covariance
    Low_triangular_Covariance=Cholesky_of_Covariance_matrix(time_array, jnp.zeros_like(time_array),
                                                                 Correlation_time, Reverberation_time, Curve_type)

    array_shape=time_array.shape

    @jax.jit
    def GP_sampler(Mean, Variance, random_seed):
        """

        Parameters
        ----------
        Mean: float
            Mean of a sampled point
        Variance: float
            Variance of a sampled point
        random_seed: uint32
            key of random generation of the Gaussian process

        Returns
        -------

        """
        random_key = jax.random.PRNGKey(random_seed)
        standard_normal_array=jax.random.normal(random_key,shape=array_shape)

        return Mean + jnp.sqrt(Variance)*jnp.dot(Low_triangular_Covariance,standard_normal_array)

    return GP_sampler

def predict(desired_times,time_array, values_array, errors_array, Mean, Variance,
            Correlation_time, Reverberation_time, Curve_type = 'RDRW',return_cov_matrix = False):

    one = jnp.ones_like(values_array)

    if Curve_type == 'RDRW':
        get_Covariance = lambda time_difference: RDRW_kernel(time_difference, 1.,
                                                             Correlation_time, Reverberation_time)
    elif Curve_type == 'DRW':
        # DRW is sampled from Matern Covariance with order 0.5, where
        get_Covariance = lambda time_difference: Matern_kernel(time_difference, 1., Correlation_time, 0.5)
    else:
        raise ValueError('Wrong light curve type. Pick one of "RDRW" or "DRW"')

    time_difference = desired_times[None,:] - time_array[:,None]

    Covariances = get_Covariance(time_difference)

    Low_triangular_Covariance =  Cholesky_of_Covariance_matrix(time_array, (errors_array ** 2) / Variance,
                                                                   Correlation_time, Reverberation_time, Curve_type)

    # Mean prediction
    alpha = (jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), values_array - Mean * one))

    predicted_mean = Mean + Covariances.T @ alpha
    # f = self.mu + k.T @ self.inv_K @ (self.y-self.mu*one)

    V = jax.scipy.linalg.solve_triangular(Low_triangular_Covariance,Covariances,lower=True,check_finite=False)

    err = (errors_array ** 2).mean()

    if return_cov_matrix:
        desired_time_difference = desired_times[None,:] - desired_times[:,None]
        predicted_corr_matrix = get_Covariance(desired_time_difference) - V.T @ V
        predicted_cov_matrix = Variance * predicted_corr_matrix  + np.eye(len(desired_times))*err
        return np.array(predicted_mean),np.array(predicted_cov_matrix)
    else:
        predicted_covariance =  get_Covariance(0.) - jnp.einsum("ij,ji->i", V.T, V)
        predicted_variance = Variance * predicted_covariance + err
        return np.array(predicted_mean),np.array(predicted_variance)

def alpha(time_array, values_array, errors_array, Mean, Variance,
            Correlation_time, Reverberation_time, Curve_type = 'RDRW'):


    one = jnp.ones_like(values_array)

    if Curve_type == 'RDRW':
        get_Covariance = lambda time_difference: RDRW_kernel(time_difference, 1.,
                                                             Correlation_time, Reverberation_time)
    elif Curve_type == 'DRW':
        # DRW is sampled from Matern Covariance with order 0.5, where
        get_Covariance = lambda time_difference: Matern_kernel(time_difference, 1., Correlation_time, 0.5)
    else:
        raise ValueError('Wrong light curve type. Pick one of "RDRW" or "DRW"')

    time_difference = time_array[None,:] - time_array[:,None]

    Covariances = get_Covariance(time_difference)

    Low_triangular_Covariance =  Cholesky_of_Covariance_matrix(time_array, (errors_array ** 2) / Variance,
                                                                   Correlation_time, Reverberation_time, Curve_type)

    alpha = (jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), values_array - Mean * one))

    norm = np.sum(jnp.log(jnp.diag(Low_triangular_Covariance))) - len(values_array)/2 * np.log(2*np.pi)

    chi = values_array * alpha

    return alpha,norm

#def chi(values_array,al)