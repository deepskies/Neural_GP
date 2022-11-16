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

from typing import Union
from . import RDRW,RDRW_optimization


import jax
import jax.numpy as jnp
from Modules.Gaussian_process import RDRW,RDRW_optimization
from typing import Union

eps = 1e-8

# TODO: write as parent-children wrp fixed parameters
class Optimizer():

    def __init__(self,time_array,value_array,errors_array,Correlation_time = None,Reverberation_time = None,
                 normalisation: Union['day','year'] = 'day',report_learning = False):

        self.normalisation = normalisation

        if normalisation=='day':
            self.times = time_array / 365
        else:
            self.times = time_array

        self.values = value_array
        self.errors = errors_array

        self.Mean = None
        self.Variance = None

        if (Correlation_time is None):
            self.optimized_task = 'All free'
            # default
            Correlation_time = np.power(10, 2.6)
            Reverberation_time = np.power(10,1.25)

        elif (Reverberation_time is None):
            self.optimized_task = 'Fixed tau'
            Reverberation_time = np.power(10,1.25)
        else:
            self.optimized_task = 'Fixed kernel'



        self.Correlation_time = Correlation_time /365
        self.Reverberation_time = Reverberation_time /365


        self.report_learning = report_learning


    def _args_to_labels(self,Arguments):
        Mean = Arguments[0]

        if self.optimized_task == 'All free':
            Correlation_time = np.power(10, Arguments[2])
            Reverberation_time = np.power(10, Arguments[3])
        elif self.optimized_task == 'Fixed tau':
            Correlation_time = self.Correlation_time
            Reverberation_time = np.power(10, Arguments[2])
        else:
            Correlation_time = self.Correlation_time
            Reverberation_time = self.Reverberation_time

        Variance = np.power(10, Arguments[1]) * (Correlation_time + Reverberation_time)

        Labels = [Mean, Variance, Correlation_time, Reverberation_time]


        return np.array(Labels)

    @partial(jax.jit, static_argnums=(0,))
    def Full_Loss_RDRW(self, Arguments):

        Mean, Descaled_logVariance = Arguments[:2]

        log_Correlation_time, log_Reverberation_time = Arguments[2:4]

        Correlation_time = jnp.power(10, log_Correlation_time)
        Reverberation_time = jnp.power(10, log_Reverberation_time)
        Variance = jnp.power(10, Descaled_logVariance) * (Correlation_time + Reverberation_time)

        NLL = RDRW_optimization.NegLogLikelihood(self.times, self.values, self.errors, Mean, Variance,
                                                 Correlation_time, Reverberation_time, Curve_type='RDRW',
                                                 Normalised=True)

        return NLL

    @partial(jax.jit, static_argnums=(0,))
    def Fixed_kernel_Loss_RDRW(self, Arguments):

        Mean, Descaled_logVariance = Arguments[:2]

        Correlation_time = self.Correlation_time
        Reverberation_time = self.Reverberation_time
        Variance = jnp.power(10, Descaled_logVariance) * (Correlation_time + Reverberation_time)

        NLL = RDRW_optimization.NegLogLikelihood(self.times, self.values, self.errors, Mean, Variance,
                                                 Correlation_time, Reverberation_time, Curve_type='RDRW',
                                                 Normalised=True)

        return NLL

    @partial(jax.jit, static_argnums=(0,))
    def Fixed_Tau_Loss_RDRW(self, Arguments):

        Mean, Descaled_logVariance,log_Reverberation_time = Arguments[:3]

        Correlation_time = self.Correlation_time
        Reverberation_time = jnp.power(10, log_Reverberation_time)
        Variance = jnp.power(10, Descaled_logVariance) * (Correlation_time + Reverberation_time)

        NLL = RDRW_optimization.NegLogLikelihood(self.times, self.values, self.errors, Mean, Variance,
                                                 Correlation_time, Reverberation_time, Curve_type='RDRW',
                                                 Normalised=True)

        return NLL

    def compile_loss(self):

        if self.optimized_task == 'All free':
            return self.Full_Loss_RDRW
        elif self.optimized_task == 'Fixed tau':
            return self.Fixed_Tau_Loss_RDRW
        else:
            return self.Fixed_kernel_Loss_RDRW

    def _prepare_for_optimization(self):

        Mean = self.values.mean()


        Correlation_time = self.Correlation_time
        Reverberation_time = self.Reverberation_time

        Descaled_Variance = self.values.var() / (Correlation_time + Reverberation_time)

        initial_RDRW_guess = np.array([Mean, jnp.log10(Descaled_Variance),
                                       jnp.log10(Correlation_time), jnp.log10(Reverberation_time)])

        Mean_bounds = [self.values.min(), self.values.max()]
        # log_Corr_time_bounds = [np.log10(np.diff(time_array).min()),np.log10(time_array.max())] # from https://iopscience.iop.org/article/10.1088/0004-637X/698/1/895/pdf
        log_Corr_time_bounds = [1.8 - np.log10(365) + eps, np.inf]
        log_Rev_time_bounds = [-np.inf, 1.8 - np.log10(365) - eps]

        lower_bound = np.array((Mean_bounds[0], -np.inf, log_Corr_time_bounds[0], log_Rev_time_bounds[0]))
        upper_bound = np.array((Mean_bounds[1], np.inf, log_Corr_time_bounds[1], log_Rev_time_bounds[1]))

        #bounds_RDRW = scipy.optimize.Bounds(lb=lower_bound,
        #                                    ub=upper_bound, keep_feasible=True)

        return initial_RDRW_guess,lower_bound,upper_bound

    def optimize(self,initial_guess = None, lower_bound = None ,upper_bound = None, **kwargs):

        suggestions = self._prepare_for_optimization()

        if initial_guess is None:
            initial_guess = suggestions[0]

        if lower_bound is None:
            lower_bound = suggestions[1]

        if upper_bound is None:
            upper_bound = suggestions[2]


        if self.optimized_task == 'All free':
            bounds = scipy.optimize.Bounds(lb=lower_bound,
                                            ub=upper_bound, keep_feasible=True)
        elif self.optimized_task == 'Fixed tau':
            initial_guess = initial_guess[[0,1,3]]
            bounds = scipy.optimize.Bounds(lb=lower_bound[[0,1,3]],
                                            ub=upper_bound[[0,1,3]], keep_feasible=True)
        else:
            initial_guess = initial_guess[:-2]
            bounds = scipy.optimize.Bounds(lb=lower_bound[:-2],
                                            ub=upper_bound[:-2], keep_feasible=True)


        Loss_RDRW = self.compile_loss()
        grad = jax.grad(Loss_RDRW)
        hess = jax.jacfwd(jax.jacrev(Loss_RDRW))

        result, learning_curve = self._optimize(Loss_RDRW,grad,hess,initial_guess,bounds,**kwargs)

        labels = self._args_to_labels(result)
        labels[2:] = labels[2:] * 365

        return labels,learning_curve



    def _optimize(self,Loss, Gradient, Hessian, guess, bounds, method='TNC', options=None, use_hessian=False):

        if options is None:
            # options = {'disp':True,'maxiter':500}
            options = {}

        learning_curve = []

        global step
        step = 1

        def callbackF(Xi, *args):
            global step
            if self.report_learning:
                print(step , np.array(Xi))
            loss = Loss(Xi)

            learning_curve.append([*Xi, loss.item()])

            step += 1


        hess = lambda x: np.array(Hessian(x))
        if not use_hessian:
            hess = None

        def loss(x):
            l = Loss(x)
            if self.report_learning:
                print(x,l)
            return l

        grad = lambda x: np.array(Gradient(x))

        res = scipy.optimize.minimize(loss, guess, method=method, jac=grad, hess=hess,
                                      bounds=bounds, options=options)

        return res.x, learning_curve

def compile_Losses(time_array,Light_curve,Light_curve_errs,Curve_type: Union['DRW','RDRW'] = 'RDRW'):

    @jax.jit
    def Loss_RDRW(Arguments):

        Mean,Descaled_logVariance,log_Correlation_time,log_Reverberation_time, Noise_std = Arguments

        Correlation_time = jnp.power(10,log_Correlation_time)
        Reverberation_time = jnp.power(10,log_Reverberation_time)
        #Variance=jnp.power(10,Descaled_logVariance)*Correlation_time/4
        Variance = jnp.power(10, Descaled_logVariance)
        NLL= RDRW_optimization.NegLogLikelihood(time_array,Light_curve,Light_curve_errs,Mean,Variance,Correlation_time,Reverberation_time,Noise_std,Curve_type='RDRW',Normalised=True)

        return NLL

    @jax.jit
    def Loss_DRW(Arguments):

        Mean,Descaled_logVariance,log_Correlation_time,Noise_std = Arguments

        Correlation_time = jnp.power(10,log_Correlation_time)
        #Variance=jnp.power(10,Descaled_logVariance)*Correlation_time/4
        Variance = jnp.power(10, Descaled_logVariance)

        NLL= RDRW_optimization.NegLogLikelihood(time_array,Light_curve,Light_curve_errs,Mean,Variance,Correlation_time,1.,Curve_type='DRW',Normalised=True)

        return NLL

    return Loss_DRW,Loss_RDRW

def prepare_for_optimization(time_array,Light_curve,Light_curve_errs):

    Mean = Light_curve.mean()
    Correlation_time = 200
    Reverberation_time = 0.01
    #Descaled_Variance = Light_curve.var() * 4 / (Correlation_time)
    Descaled_Variance = Light_curve.var()

    amplitude = Light_curve.max()-Light_curve.min()

    initial_guess = [Mean,jnp.log10(Descaled_Variance),
                     jnp.log10(Correlation_time),jnp.log10(Reverberation_time),0.01 * amplitude]

    Mean_bounds = [Light_curve.min(),Light_curve.max()]
    log_Corr_time_bounds = [np.log10(np.diff(time_array).min()),3.8] # from https://iopscience.iop.org/article/10.1088/0004-637X/698/1/895/pdf
    log_Rev_time_bounds = [-np.inf,2.5] # from https://iopscience.iop.org/article/10.3847/1538-4357/ab3728/pdf


    bounds_DRW=scipy.optimize.Bounds(lb = np.array((Mean_bounds[0],-np.inf,log_Corr_time_bounds[0], 0. )),
                             ub = np.array((Mean_bounds[1],np.inf, 3.4, amplitude)), keep_feasible=True)

    bounds_RDRW=scipy.optimize.Bounds(lb = np.array((Mean_bounds[0],-np.inf,log_Corr_time_bounds[0], log_Rev_time_bounds[0],0. )),
                                      ub = np.array((Mean_bounds[1],np.inf, log_Corr_time_bounds[1], log_Rev_time_bounds[1],amplitude)), keep_feasible=True)


    def constraint_RDRW(Arguments):
        # correlation time, higher than reverberation time
        return Arguments[-2] - Arguments[-1]

    Loss_DRW,Loss_RDRW = compile_Losses(time_array,Light_curve,Light_curve_errs)
    DRW_grad = jax.grad(Loss_DRW)
    DRW_hess = jax.jacfwd(jax.jacrev(Loss_DRW))

    RDRW_grad = jax.grad(Loss_RDRW)
    RDRW_hess = jax.jacfwd(jax.jacrev(Loss_RDRW))

    return [Loss_DRW,DRW_grad,DRW_hess,initial_guess[:-1],bounds_DRW],\
           [Loss_RDRW,RDRW_grad,RDRW_hess,initial_guess,bounds_RDRW,constraint_RDRW]

def optimize(Loss,Gradient,Hessian,guess,bounds,constraint = None,method = 'trust-constr', options = None, use_hessian = True):

    if options is None:
        options = {'disp':True,'maxiter':500}

    learning_curve= []

    def callbackF(Xi,*args):
        loss=Loss(Xi)

        learning_curve.append([*Xi,loss.item()])

    if not use_hessian:
        Hessian = None

    if constraint is None:
        res=scipy.optimize.minimize(Loss,guess,method=method,jac=Gradient,hess=Hessian,
                                    bounds=bounds,callback=callbackF,options=options)
    else:
        res = scipy.optimize.minimize(Loss, guess, method=method, jac=Gradient, hess=Hessian,
                                      bounds=bounds,callback=callbackF, options=options, constraints = {'type': 'ineq', 'fun': constraint})

    return res.x,np.stack(learning_curve)

def args_to_labels(Arguments):

    Mean = Arguments[0]
    Correlation_time = jnp.power(10,Arguments[2])
    #Variance=jnp.power(10,Arguments[1])*Correlation_time/4
    Variance = jnp.power(10, Arguments[1])
    Noise_std = Arguments[-1]

    Labels = np.array([Mean,Variance,Correlation_time])

    if len(Arguments)==4:
        Reverberation_time = jnp.power(10,Arguments[3])
        Labels = np.append(Labels,Reverberation_time)

    Labels = np.append(Labels,Noise_std)

    return Labels

def labels_to_args(Labels):

    Mean = Labels[0]
    log_Correlation_time = jnp.log10(Labels[2])
    #Descaled_logVariance = jnp.log10(Labels[1] * 4 /Labels[2])
    Descaled_logVariance = jnp.log10(Labels[1])
    Noise_std = Labels[-1]

    args = np.array([Mean,Descaled_logVariance,log_Correlation_time])

    if len(Labels)==4:
        log_Reverberation_time = jnp.log10(Labels[3])
        args = np.append(args,log_Reverberation_time)

    args = np.append(args, Noise_std)
    return args