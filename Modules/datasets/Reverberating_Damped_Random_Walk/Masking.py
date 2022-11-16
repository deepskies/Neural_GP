import numpy as np
from .Utils import Tensor_Dataset
from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# getter for sampler of truncated normal distribution (sample offset bounds of season gaps as offset from 365*n)
def get_truncnorm_sampler(mean,std,lower_limit,upper_limit,rng):
    normalised_limits = (lower_limit - mean) / std , (upper_limit - mean) / std

    def sampler(size):
        return stats.truncnorm.rvs(*normalised_limits, size=size, loc=mean, scale=std,random_state=rng)

    return sampler

def Poisson_process_subsampling_mask(process_rate, desired_rate, shape, rng = None):
    """
    Theorem 1.4 (Partitioning a Poisson process)
    If $\psi \sim PP(\lambda)$ and if each arrival of $\psi$ is, independently, type 1 or type 2
    with probability $p$ and $q = 1 − p$ then in fact, letting $\psi_i$ denote the point process
    of type $i$ arrivals, $i = 1, 2$, $ψ_1 \sim PP(p\lambda)), ψ_2 \sim PP(q\lambda))$ and they are independent.

    @inproceedings{Sigman20061I6,
    title={1 IEOR 6711 : Notes on the Poisson Process},author={Karl Sigman},year={2006}
    }
    """
    assert desired_rate<=process_rate

    if rng is None:
        rng = np.random.default_rng(seed=42)

    subsampling_probability = desired_rate/process_rate

    mask = rng.binomial(1,subsampling_probability,size=shape).astype(bool)

    return mask



def sample_border_masks(time_arrays,gap_sampler):
    """
    Parameters
    ----------
    time_arrays: array (M,N)
        time arrays normalised to
    gap_sampler: function() -> float
        samples length of border gaps in days/365
    Returns
    -------
     start_gap_mask,end_gap_mask: mask covering begining and end of observations

    """

    gaps_lengths = gap_sampler(2 * time_arrays.shape[0]).reshape(2, -1)

    # start and end of the season (in normalized year coordinates)
    gaps_borders = np.array([gaps_lengths[0], 1 - gaps_lengths[1]]).T

    start_gap_mask = (time_arrays < gaps_borders[:, None, 0])
    end_gap_mask = (time_arrays > gaps_borders[:, None, 1])

    return start_gap_mask,end_gap_mask