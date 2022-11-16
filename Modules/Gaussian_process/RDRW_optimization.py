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


import jaxlib

# for Matern kernel with Modified Bessel of the 2nd kind (tfp>=0.17.0-dev20220322)
import tensorflow_probability as tfp


#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(max_thread_numbers)

# This notebook is meant to handle generation of Reverberating Damped Random Walk Gaussian-Markov process

from . import RDRW

#softening
eps = 1e-8

@partial(jax.jit, static_argnums=(7, 8))
def NegLogLikelihood(time_array: jnp.ndarray, values_array: jnp.ndarray, errors_array: jnp.ndarray,
                     Mean: float, Variance: float, Correlation_time: float,
                     Reverberation_time:float, Curve_type='RDRW', Normalised=True):
    """
    Estimates negative log-likelihood of hypothesis that a given time series (t,v) with errors e
    is a gaussian process with the parameters Mean,Variance,Correlation_time,Reverberation_time
    Parameters
    ----------
    time_array: array (N,) [days]
        time stamps, on which the process should be sampled
    values_array: array (N,) [days]
        values observed in the time points
    errors_array: array (N,) [days]
        uncertainties on those values expressed as gaussian standard deviation
    Mean: float
        Mean of the Gaussian process
    Variance: float
        Variance of the Gaussian process
    Correlation_time: float [days]
        Measure of global correlation between separated points. Defines global evolution.
    Reverberation_time: float [days]
        Measure of local correlation between separated points. Defines local evolution.
        It is assumed that Reverberation_time<Correlation_time
    Noise_std: float
        Std of noise atributed to the data
    Curve_type: 'RDRW' or 'DRW'
        Assumes either Reverberating Damped Random Walk (RDRW) kernel or DRW kernel (Matern[0.5])
    Normalised: bool
        True to include factor normalising total probability from the PDF to 1. False to consider 1/2*chi^2

    Returns
    -------
        -lnp(mu,sigma^2,t_corr,t_rev|t,x,e): float
    """

    N=len(time_array)


    # Use Cholesky decomposition to get lower triangle matrix describing covariance
    Low_triangular_Covariance = RDRW.Cholesky_of_Covariance_matrix(time_array, (errors_array ** 2) / Variance,
                                                                   Correlation_time, Reverberation_time, Curve_type)

    # Mean estimation mu = (one.T @ inv_K @ self.y)/ (one.T @ inv_K @ one)
    one = jnp.ones_like(values_array)

    # Euclidian Error estimation (self.y-mu*one).T @ inv_K @ (self.y-mu*one)
    Squared_Error = (values_array - Mean * one).T @ (jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), values_array - Mean * one))
    Chi_squared = Squared_Error / (Variance+eps) / 2

    if not Normalised:
        return Chi_squared
    else:
        # log(det(K))
        LnDetCov = 2 * jnp.sum(jnp.log(jnp.diag(Low_triangular_Covariance)))
        # Normalisation of multivariate gaussian distribution
        Normalisation = 0.5 * N * (np.log(2*np.pi) + jnp.log(Variance)) + 0.5 * LnDetCov
        return Normalisation + Chi_squared
    
def get_NLL_sampler(time_array: jnp.ndarray, values_array: jnp.ndarray, Correlation_time: float,
                    Reverberation_time:float, Curve_type='RDRW', Normalised=True):
    """
    Precompiles function returns a function F[t_corr,t_rev](Mean,Variance) that estimates negative log-likelihood of hypothesis
    that a given time series (t,v) is a gaussian process with the parameters Mean,Variance,t_corr,t_rev
    
    Parameters
    ----------
    time_array: array (N,) [days]
        time stamps, on which the process should be sampled
    values_array: array (N,) [days]
        values observed in the time points
    Correlation_time: float [days]
        Measure of global correlation between separated points. Defines global evolution.
    Reverberation_time: float [days]
        Measure of local correlation between separated points. Defines local evolution.
        It is assumed that Reverberation_time<Correlation_time
    Curve_type: 'RDRW' or 'DRW'
        Assumes either Reverberating Damped Random Walk (RDRW) kernel or DRW kernel (Matern[0.5])
    Normalised: bool
        True to include factor normalising total probability from the PDF to 1. False to consider 1/2*chi^2

    Returns
    -------
        -lnp(mu,sigma^2,t_corr,t_rev|t,x,e): float
    """


    N=len(values_array)

    # Use Cholesky decomposition to get lower triangle matrix describing covariance
    Low_triangular_Covariance = RDRW.Cholesky_of_Covariance_matrix(time_array,np.zeros(N),
                                                                             Correlation_time, Reverberation_time,
                                                                             Curve_type)

    # Mean estimation mu = (one.T @ inv_K @ self.y)/ (one.T @ inv_K @ one)
    one = jnp.ones_like(values_array)

    @jax.jit
    def NLL_sampler(Mean, Variance):
        """
        Mean: float
            Mean of the Gaussian process
        Variance: float
            Variance of the Gaussian process
        Returns
        -------
        -lnp(mu,sigma^2,t_corr,t_rev|t,x): float
        """

        # Euclidian Error estimation (self.y-mu*one).T @ inv_K @ (self.y-mu*one)
        Squared_Error = (values_array - Mean * one).T @ (
            jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), values_array - Mean * one))
        Chi_squared = Squared_Error / (Variance + eps) / 2

        if not Normalised:
            return Chi_squared
        else:
            # log(det(K))
            NegLog_Covariance_determinant = 2 * jnp.sum(jnp.log(jnp.diag(Low_triangular_Covariance)))
            # Normalisation of multivariate gaussian distribution
            Neglog_Normalisation_factor = 0.5 * N * (np.log(2 * np.pi) + jnp.log(Variance)) + 0.5 * NegLog_Covariance_determinant
            return Neglog_Normalisation_factor + Chi_squared

    return NLL_sampler


@partial(jax.jit, static_argnums=(0, 1))
def deshift_time_series(time_array: tuple, values_arrays: tuple,
                        values_shifts: jnp.ndarray, time_shifts: jnp.ndarray):



    # Shift time arrays except for the first one
    time_array = np.array(time_array)
    delayed_time_arrays = jnp.array([time_array + time_delay for time_delay in time_shifts])
    delayed_time_arrays = jnp.append(time_array.reshape((1,-1)), delayed_time_arrays, axis=0)

    # Shift Light curves except for the first one
    values_arrays=np.array(values_arrays)
    deshifted_values_arrays = jnp.array([values_arrays[i + 1] + shift for i, shift in
                                          enumerate(values_shifts)])
    deshifted_values_arrays = jnp.append(values_arrays[0].reshape((1, -1)), deshifted_values_arrays, axis=0)

    return delayed_time_arrays, deshifted_values_arrays

@partial(jax.jit, static_argnums=(0, 1, 2, 9))
def NegLogLikelihood_multiseries(time_array: tuple, values_arrays: tuple, errors_arrays: tuple,
                                 Mean: float, Variance: float, Correlation_time: float, Reverberation_time: float,
                                 values_shifts: jnp.ndarray, time_shifts: jnp.ndarray,Curve_type='RDRW', Normalised=True):
    """
    Estimates negative log-likelihood of hypothesis that a given time series (t,v) with errors e
    is a gaussian process with the parameters Mean,Variance,Correlation_time,Reverberation_time
    Parameters
    ----------
    time_array: array (N,) [days]
        time stamps, on which the process should be sampled
    values_arrays: array (M,N) [days]
        values of several time series observed in the same time points,
        that are created by shifting some parts of the original series in time and magnitude
    errors_arrays: array (M,N) [days]
        uncertainties of several time series observed in the same time points expressed as gaussian standard deviation
    Mean: float
        Mean of the Gaussian process
    Variance: float
        Variance of the Gaussian process
    Correlation_time: float [days]
        Measure of global correlation between separated points. Defines global evolution.
    Reverberation_time: float [days]
        Measure of local correlation between separated points. Defines local evolution.
        It is assumed that Reverberation_time<Correlation_time
    Curve_type: 'RDRW' or 'DRW'
        Assumes either Reverberating Damped Random Walk (RDRW) kernel or DRW kernel (Matern[0.5])
    Normalised: bool
        True to include factor normalising total probability from the PDF to 1. False to consider 1/2*chi^2

    Returns
    -------
        -lnp(mu,sigma^2,t_corr,t_rev|t,x,e): float
    """

    delayed_time_arrays, demagnified_light_curves=deshift_time_series(time_array, values_arrays, values_shifts, time_shifts)

    NLL=NegLogLikelihood(delayed_time_arrays.flatten(), demagnified_light_curves.flatten(), np.array(errors_arrays).flatten(),
                                      Mean, Variance, Correlation_time, Reverberation_time, Curve_type = Curve_type, Normalised=Normalised)

    return NLL

def NegLogPosterior(Arguments, NegLogLikelihood_pure_function: jaxlib.xla_extension.CompiledFunction,
                    NegLogPrior_pure_function: jaxlib.xla_extension.CompiledFunction, Sanity_check_pure_function = None):

    # Compute priors for given Arguments
    NegLogPrior = NegLogPrior_pure_function(Arguments)

    if Sanity_check_pure_function is None:
        NegLogPosterior= NegLogLikelihood_pure_function(Arguments)+NegLogPrior
    elif isinstance(Sanity_check_pure_function,jaxlib.xla_extension.CompiledFunction):
        # If values are sane, call NegLogLikelihood and return NegLogPosterior, otherwise return only NegLogPrior
        NegLogPosterior=jax.lax.cond(Sanity_check_pure_function(Arguments),
                                     lambda Arguments: NegLogLikelihood_pure_function(Arguments) + NegLogPrior,
                                     lambda _: NegLogPrior, Arguments)
    else:
        raise ValueError('Sanity check should be either None or jax-compiled function')

    return NegLogPosterior

@partial(jax.jit, static_argnums=(0, 4))
def Cholesky_of_multicurve_Covariance_matrix(time_array: tuple, Correlation_time: float, Reverberation_time: float, time_shifts: jnp.ndarray, Curve_type='RDRW'):

    time_array = np.array(time_array)
    delayed_time_arrays = jnp.array([time_array + time_delay for time_delay in time_shifts])
    delayed_time_arrays = jnp.append(time_array.reshape((1, -1)), delayed_time_arrays, axis=0)

    N = delayed_time_arrays.size

    # Use Cholesky decomposition to get lower triangle matrix describing covariance
    Low_triangular_Covariance = RDRW.Cholesky_of_Covariance_matrix(delayed_time_arrays.flatten(), np.zeros(N), Correlation_time, Reverberation_time, Curve_type)

    return Low_triangular_Covariance,delayed_time_arrays

@partial(jax.jit, static_argnums=(2,))
def MaxLikelihood_estimators_Mean_Magnifications(Low_triangular_Covariance, values_arrays, num_of_arrays):
    """
    Case num_of_curves=1:
        mu= 1^T K^{-1} y_A / 1^T K^{-1} 1

    Case num_of_curves=2:

        dM_B=(1,0)^T K^{-1} (y_B-y_A, y_A-y_B) / (1,0)^T K^{-1} (1,-1)
        mu= (1,1)^T K^{-1} (y_A, y_B - dM_B) / (1,1)^T K^{-1} (1,1)

    case num_of_curves=3:
        AC * x=(1,0,-1)^T K^{-1} (x,0,0)
        DC * x=(-1,1,0)^T K^{-1} (0,0,x)
        AB * x=(1,-1,0)^T K^{-1} (x,0,0)
        DB * x=(-1,0,1)^T K^{-1} (0,x,0)

        dM_B=((AC*1,- DC*1)^T * (AB * (y_B - y_A) + DC * y_C, AC * (y_C - y_A) + DB * y_B)) /
                ((AC*1,- DC*1)^T * (AB * 1, DB * 1))
        dM_C = // exchange places of B and C //

        mu = (1,1,1)^T K^{-1} (y_A, y_B - dM_B, y_C - dM_C) / (1,1,1)^T K^{-1} (1,1,1)

    case num_of_curves=4:


    Parameters
    ----------
    Low_triangular_Covariance: Cholesky decomposition of the Covariance matrix (normalised by Variance)
    num_of_arrays

    Returns
    -------

    """

    if num_of_arrays == 1:
        one = jnp.ones(len(Low_triangular_Covariance))

        Mean = one.T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), values_arrays.flatten())
        Mean = Mean / (one.T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), one))

        deshifted_values_arrays = values_arrays
        Mean_and_Magnifications = [Mean]

    elif num_of_arrays == 2:
        one = jnp.ones(len(Low_triangular_Covariance) // 2)
        zero = jnp.zeros(len(Low_triangular_Covariance) // 2)
        # y_B - y_A
        LC_B_A = values_arrays[1] - values_arrays[0]

        dM_B = jnp.append(one, zero).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), jnp.append(LC_B_A, -LC_B_A))
        dM_B = dM_B / (
                    jnp.append(one, zero).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), jnp.append(one, -one)))

        # Change sign of magnification to keep to the convention
        dM_B = -dM_B
        deshifted_values_arrays = jnp.array([values_arrays[0], values_arrays[1] + dM_B])

        Mean = jnp.append(one, one).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), deshifted_values_arrays.flatten())
        Mean = Mean / (jnp.append(one, one).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), jnp.append(one, one)))

        Mean_and_Magnifications = [Mean, dM_B]


    elif num_of_arrays == 3:

        one = jnp.ones(len(Low_triangular_Covariance) // 3)
        zero = jnp.zeros(len(Low_triangular_Covariance) // 3)

        def Bil_form(Matrix, Vector):
            """
            Bilinear form for given vector

            AC * x=(1,0,-1)^T K^{-1} (x,0,0) = 1^T (a-c) x
            DC * x=(-1,1,0)^T K^{-1} (0,0,x) = 1^T (d-c) x
            AB * x=(1,-1,0)^T K^{-1} (x,0,0) = 1^T (a-b) x
            DB * x=(-1,0,1)^T K^{-1} (0,x,0) = 1^T (d-b) x
            """

            if Matrix == 'AC':
                return jnp.append(one, zero, -one).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True),
                                                                                  jnp.append(Vector, zero, zero))
            elif Matrix == 'DC':
                return jnp.append(-one, one, zero).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True),
                                                                                  jnp.append(zero, zero, Vector))
            elif Matrix == 'AB':
                return jnp.append(one, -one, zero).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True),
                                                                                  jnp.append(Vector, zero, zero))
            elif Matrix == 'DB':
                return jnp.append(-one, zero, one).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True),
                                                                                  jnp.append(zero, Vector, zero))
            else:
                raise ValueError('Wrong Bilinear form type. Pick one of "AC","DC","AB","DB"')

        """
        dM_B=((AC*1,- DC*1)^T * (AB * (y_B - y_A) + DC * y_C, AC * (y_C - y_A) + DB * y_B)) /
                ((AC*1,- DC*1)^T * (AB * 1, DB * 1))
        dM_C = // exchange places of B and C //

        mu = (1,1,1)^T K^{-1} (y_A, y_B - dM_B, y_C - dM_C) / (1,1,1)^T K^{-1} (1,1,1)
        """

        dM_B = jnp.append(Bil_form('AC', one), -Bil_form('DC', one)).T @ jnp.append(
            Bil_form('AB', values_arrays[1] - values_arrays[0]) + Bil_form('DC', values_arrays[2]),
            Bil_form('AC', values_arrays[2] - values_arrays[0]) + Bil_form('DB', values_arrays[1]))
        dM_B = dM_B / (jnp.append(Bil_form('AC', one), -Bil_form('DC', one)).T @ jnp.append(Bil_form('AB', one),
                                                                                            -Bil_form('DB', one)).T)

        # Same but B<->C
        dM_C = jnp.append(Bil_form('AB', one), -Bil_form('DB', one)).T @ jnp.append(
            Bil_form('AC', values_arrays[2] - values_arrays[0]) + Bil_form('DB', values_arrays[1]),
            Bil_form('AB', values_arrays[1] - values_arrays[0]) + Bil_form('DC', values_arrays[2]))
        dM_C = dM_C / (jnp.append(Bil_form('AB', one), -Bil_form('DB', one)).T @ jnp.append(Bil_form('AC', one),
                                                                                            -Bil_form('DC', one)).T)

        # Change sign of magnification to keep to the convention
        dM_B = -dM_B
        dM_C = -dM_C

        deshifted_values_arrays = jnp.array([values_arrays[0], values_arrays[1] + dM_B, values_arrays[2] + dM_C])

        Mean = jnp.append(one, one, one).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True), deshifted_values_arrays.flatten())
        Mean = Mean / (jnp.append(one, one, one).T @ jax.scipy.linalg.cho_solve((Low_triangular_Covariance, True),
                                                                                jnp.append(one, one, one)))

        Mean_and_Magnifications = [Mean, dM_B, dM_C]

    elif num_of_arrays == 4:
        raise NotImplementedError

    else:
        raise ValueError(
            'We consider num_of_curves from 1 to 4 light curves as a usual limit of Lensed quasar images number')

    return Mean_and_Magnifications,deshifted_values_arrays


@partial(jax.jit, static_argnums=(2,))
def MaxLikelihood_estimators_Gauss_process(values_arrays, Low_triangular_Covariance, num_of_curves):

    Mean_and_Magnifications, deshifted_values_arrays = MaxLikelihood_estimators_Mean_Magnifications(Low_triangular_Covariance, values_arrays, num_of_curves)

    Mah_distance = RDRW.Mahalanobis_distance(deshifted_values_arrays.flatten(),Mean_and_Magnifications[0],Low_triangular_Covariance)
    Variance = Mah_distance**2/len(Low_triangular_Covariance)

    return Variance,Mean_and_Magnifications,deshifted_values_arrays

@partial(jax.jit, static_argnums=(0, 1, 4))
def NegLogLikelihood_singlecurve_MLE_based(time_array: tuple, values_array: tuple,
                                           Correlation_time: float, Reverberation_time: float,
                                           Curve_type='RDRW'):

    time_array=np.array(time_array)
    N = time_array.size

    # Use Cholesky decomposition to get lower triangle matrix describing covariance
    Low_triangular_Covariance = RDRW.Cholesky_of_Covariance_matrix(time_array.flatten(),
                                                                             np.zeros(N),
                                                                             Correlation_time, Reverberation_time,
                                                                             Curve_type)

    MLE = MaxLikelihood_estimators_Gauss_process(Low_triangular_Covariance, np.array(values_array), 1.)
    Variance_MLE = MLE[0]

    # log(det(K))
    NegLog_Covariance_determinant = 2 * jnp.sum(jnp.log(jnp.diag(Low_triangular_Covariance)))
    # Normalisation of multivariate gaussian distribution
    Neglog_Normalisation_factor = 0.5 * N * (
                np.log(2 * np.pi) + jnp.log(Variance_MLE)) + 0.5 * NegLog_Covariance_determinant

    # Normalisation factor and exponent in max likelihood point
    return Neglog_Normalisation_factor + N / 2

@partial(jax.jit, static_argnums=(0, 1, 4))
def Singlecurve_MLE_Statistics(time_array: tuple, values_array: tuple,
                               Correlation_time: float, Reverberation_time: float,
                               Curve_type='RDRW'):

    time_array = np.array(time_array)
    N = time_array.size

    # Use Cholesky decomposition to get lower triangle matrix describing covariance
    Low_triangular_Covariance = RDRW.Cholesky_of_Covariance_matrix(time_array.flatten(), np.zeros(N), Correlation_time, Reverberation_time, Curve_type)

    values_array = np.array(values_array)
    Variance, Mean_and_Magnifications, _ = MaxLikelihood_estimators_Gauss_process(values_array, Low_triangular_Covariance, 1.)

    return Low_triangular_Covariance,Variance,Mean_and_Magnifications

@partial(jax.jit, static_argnums=(0, 1, 5))
def Multicurve_MLE_Statistics(time_array: tuple, values_arrays: tuple,
                              Correlation_time: float, Reverberation_time: float, time_shifts: jnp.ndarray,
                              Curve_type='RDRW'):

    Low_triangular_Covariance,delayed_time_arrays=Cholesky_of_multicurve_Covariance_matrix(time_array, Correlation_time, Reverberation_time, time_shifts, Curve_type)

    # Maximum likelihood estimators
    values_arrays = np.array(values_arrays)
    Variance,Mean_and_Magnifications,deshifted_values_arrays = MaxLikelihood_estimators_Gauss_process(Low_triangular_Covariance, values_arrays, len(values_arrays))


    # Normalisation factor and exponent in max likelihood point
    return Low_triangular_Covariance,delayed_time_arrays,deshifted_values_arrays,Variance,Mean_and_Magnifications
