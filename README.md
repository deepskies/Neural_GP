# Neural Inference of Gaussian Processes for Time Series Data of Quasars

## Abstract

The study of single-band quasar light curves poses two problems: 
inference of the power spectrum and interpolation of an irregularly 
sampled time series. A baseline approach to these tasks is to interpolate 
a time series with a Damped Random Walk (DRW) model, in which the spectrum 
is inferred using Maximum Likelihood Estimation (MLE). However, the DRW model
does not describe the smoothness of the time series, whereas MLE faces 
many problems from the theory of optimization and computational math. 
In this work, we introduce a new stochastic model that we call 
*Convolved Damped Random Walk* (CDRW). This model introduces a 
concept of smoothness to a Damped Random Walk, which enables it to 
describe quasar spectra completely. Moreover, we introduce a new method 
of inference of Gaussian process parameters, which we call 
*Neural inference*. This method uses the powers of state-of-the-art 
neural networks to improve the conventional MLE inference technique. 
In our experiments, the Neural inference method results in significant 
improvement over the baseline MLE 
(RMSE: 0.318 -> 0.205, 0.464 -> 0.444). 
Moreover, the combination of both the CDRW model and the Neural inference
significantly outperforms the baseline DRW and MLE in interpolating of a 
typical quasar light curve (chi squared: 0.333 -> 0.998, 
2.695 -> 0.981).

## For your information

I believe we, as a scientific community, should praise publishing the code for CS-related papers.
Otherwise, the one who wants to use the findings has to redevelop all the code from scratch. 
I wasted so much time remaking the code of the papers that appeared to be useless to me. So I do not want to make someone
pass through the same with my paper. 

"All the pain forced me to grow up" - Nagato Uzumaki

Unfortunately, I'm going through a tough time in my life right now, so I can not 
find time to make a prettified package out of this code. However, I can still publish what is left
from the experiments and save the time of someone who might want to use my work.

## Structure

These are all the notebooks needed to reproduce the paper's numbers and figures.

* Kepler_analysis
  * Kepler_data_reduction
    + **Data_reduction.ipynb** (Acquisition and reduction of Kepler's observations of AGN.)
    + **Curve_stiching.ipynb** (Stitching quarters of Kepler's observations. This notebook produces the final light curves.)
  * **Power_spectrum_example.ipynb** (Produces Figure 2 from Kepler's data)
* Neural_Network
  * **Neural_regression_experiments.ipynb** (Training of the Neural Network from Figure 3.)
  * **Performane_comparison.ipynb** (Comparison of Neural regression and Max likelihood. Produces Tables 1,2.)
* **Interpolation_example.ipynb** (Compares DRW and CDRW models by producing Figure 1.)

