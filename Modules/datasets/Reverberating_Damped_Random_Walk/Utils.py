import numpy as np
from torch.utils.data import Dataset
from typing import Union

__all__=['Tensor_Dataset','SubDataset','Normalise']

class Tensor_Dataset(Dataset):

    @property
    def shape(self):
        # This should be overwritten in child class
        raise NotImplementedError

    def get_item(self, *multi_index):

        # datapoint = self._dataset[multi_index]
        # return datapoint

        raise NotImplementedError

    @property
    def len(self):
        return np.prod(self.shape[:-1])

    def index_transform(self, idx, ravel=True):
        """transforms between raveled index and multiindex"""
        if ravel:
            return np.ravel_multi_index(idx, dims=self.shape[:-1])
        else:
            return np.unravel_index(idx, shape=self.shape[:-1])

    def __len__(self):
        """product of (labels num, time samplings num, random seeds num, season masks num)"""
        return self.len

    def __getitem__(self, index):

        if index>=self.len:
            raise IndexError

        """ Return masked data with dimensions (filters,time_dimensions). For our case only 1 filter """
        multi_index=self.index_transform(index,ravel=False)

        datapoint = self.get_item(*multi_index)

        return datapoint




class SubDataset(Dataset):

    def __init__(self,dataset: Tensor_Dataset,indices):

        self.dataset=dataset
        self.indices=indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index_in_dataset = self.indices[index]
        return self.dataset.__getitem__(index_in_dataset)

class Normalise:

    def __init__(self, light_curve_target_max: float = 1., light_curve_target_min: float = -1., time_norm: float = 365.):

        assert light_curve_target_max > light_curve_target_min
        assert (time_norm > 0)

        self._light_curve_target_max=light_curve_target_max
        self._light_curve_target_min=light_curve_target_min
        self._time_norm=time_norm

    def normalise_time(self,time_array: np.ndarray,shift=None, norm_factor = None):

        # Normalise time array to [0,max/time_norm]
        if shift is None:
            shift = np.min(time_array)
        if norm_factor is None:
            norm_factor = self._time_norm

        shifted_time_array=time_array-shift
        # Unified normalisation for all the curves
        normalised_time_array = shifted_time_array / norm_factor

        return normalised_time_array

    def light_curve_normalisation(self,curve_max,curve_min,target_max = None, target_min = None):


        if target_max is None:
            target_max = self._light_curve_target_max
        if target_min is None:
            target_min = self._light_curve_target_min


        curve_norm_factor= (target_max - target_min) / (curve_max - curve_min + 1e-10)
        curve_norm_shift= target_min - curve_min * curve_norm_factor

        return curve_norm_factor,curve_norm_shift

    def normalise_light_curve(self,light_curve: np.ndarray,curve_norm_factor = None, curve_norm_shift = None):

        if (curve_norm_factor is None) and (curve_norm_shift is None):
            curve_norm_factor, curve_norm_shift = self.light_curve_normalisation(np.max(light_curve),np.min(light_curve))

        normalised_light_curve= light_curve * curve_norm_factor + curve_norm_shift

        return normalised_light_curve

    def normalise_label(self, label: Union[np.ndarray,list], curve_norm_factor: float, curve_norm_shift: float, time_factor = None):


        if time_factor is None:
            time_factor = self._time_norm

        # Mean normalised according to Light_curve.
        # Corr_length and t_lambda normalised according to time_array
        # SF^2 ~ Variance/corr_time hence it transforms like curve_norm_factor**2/time_norm_factor
        #label_norm_factors = np.array([curve_norm_factor, (curve_norm_factor ** 2) * time_factor,
        #                               1. / time_factor, 1. / time_factor])
        # we assume that there is SF^2 in label[1]
        label_norm_factors = np.array([curve_norm_factor, curve_norm_factor ** 2,
                                       1. / time_factor, 1. / time_factor])
        label_norm_shifts= np.array([curve_norm_shift, 0., 0., 0.])
        normalised_label = label * label_norm_factors + label_norm_shifts

        return normalised_label

    def __call__(self, light_curve: np.ndarray, time_array: np.ndarray, label: Union[np.ndarray,list]):
        """

        Parameters
        ----------
        light_curve: array (d,)
        time_array: array (d,)
        label: array [Mean, SF_inf^2, Corr_length, t_lambda]
        Returns
        -------
            Normalised_Light_curve: array (d,). Values in [self.Light_curve_min,self.Light_curve_max]
            Normalised_time_array: array (d,). Values in [0,time_array.max()/self.time_norm]
            Normalised_labels: labels rescaled according to the data
        """

        normalised_time_array = self.normalise_time(time_array)

        curve_norm_factor, curve_norm_shift = self.light_curve_normalisation(np.max(light_curve), np.min(light_curve))
        normalised_light_curve = self.normalise_light_curve(light_curve)

        normalised_label =self.normalise_label(label,curve_norm_factor,curve_norm_shift)

        return normalised_light_curve, normalised_time_array, normalised_label

from numpy.core.shape_base import asanyarray,normalize_axis_index,_nx

def safe_stack(arrays, axis=0, out=None):

    shapes = asanyarray([asanyarray(arr).shape for arr in arrays])

    # usual np.stack behaviour when all dimensions are the same
    if len(np.unique(shapes))==1:
        return np.stack(arrays,axis=axis,out=out)

    common_shapes = (np.diff(shapes, axis=0) == 0).all(axis=0)
    last_consequent_common_dim = np.where(np.cumsum(common_shapes) == np.arange(1, len(common_shapes) + 1))[0][-1]
    common_shape = shapes[0][:last_consequent_common_dim + 1]

    static_arrays = np.empty((len(arrays), *common_shape), dtype=object)

    # Insert data into array without any casting into big ndarrays (it will only have the dimensions we set)
    for i, arr in enumerate(arrays):
        static_arrays[i] = arr

    axis = 1
    result_ndim = static_arrays[0].ndim + 1
    axis = normalize_axis_index(axis, result_ndim)

    sl = (slice(None),) * axis + (_nx.newaxis,)
    expanded_arrays = [arr[sl] for arr in static_arrays]

    return _nx.concatenate(expanded_arrays, axis=axis,out=out)