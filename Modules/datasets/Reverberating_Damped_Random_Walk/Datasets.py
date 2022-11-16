import numpy as np
import torch
from . import Masking
from .Utils import Tensor_Dataset,Normalise
from copy import deepcopy

class Year_Dataset(Tensor_Dataset):

    def __init__(self, light_curves: np.ndarray,time_arrays: np.ndarray, labels: np.ndarray,
                 output_padding_length = 165):

        super().__init__()
        max_times = time_arrays.max(axis=1)
        years_number = (max_times / 365).astype(int)
        self.years_number = np.min(years_number)

        # Storages
        self.labels = labels
        self.time_arrays = time_arrays
        self.light_curves = light_curves

        self.year_mask_indices = np.empty((time_arrays.shape[0],self.years_number,2),dtype=int)


        year_bounds = 365 * np.arange(1, self.years_number + 1)
        for time_array_index, time_array in enumerate(time_arrays):
            first_index = 0
            for year_index, year_time_end in enumerate(year_bounds):
                last_index = np.where(time_array<year_time_end)[0][-1]
                self.year_mask_indices[time_array_index,year_index]=[first_index,last_index]

                first_index=last_index+1

        # +2 is needed so we can augment with border values
        assert  output_padding_length>=self.seasons_lengths('pixel').max()+2
        self.output_padding_length = output_padding_length

    @property
    def shape(self):
        return (*self.light_curves.shape[:3],self.years_number)

    def get_labels(self, labels_index):
        return self.labels[labels_index]

    def get_time_array(self, time_index):
        return self.time_arrays[time_index]

    def get_light_curve(self, labels_index, time_index, random_seed_index):
        return self.light_curves[labels_index, time_index, random_seed_index]

    def get_year_time_arrays(self,time_index):

        years_masks = self._get_years_masks(time_index)
        flat_time_array = self.get_time_array(time_index)
        time_arrays = np.zeros(self.years_number, dtype=object)

        for year_number, mask in enumerate(years_masks):
            time_arrays[year_number] = flat_time_array[mask]/365 - year_number

        return time_arrays

    def get_item(self,labels_index, time_index, random_seed_index):

        years_masks = self._get_years_masks(time_index)

        flat_time_array = self.get_time_array(time_index)
        flat_light_curve = self.get_light_curve(labels_index, time_index, random_seed_index)
        label = self.get_labels(labels_index)

        time_arrays = np.zeros((self.years_number,self.output_padding_length),dtype=float)
        light_curves = np.zeros((self.years_number,self.output_padding_length),dtype=float)
        padding_masks = np.zeros((self.years_number,self.output_padding_length),dtype=bool)

        for year_number, mask in enumerate(years_masks):

            length = mask.sum()

            time_arrays[year_number,1:length+1] = flat_time_array[mask]/365 - year_number
            light_curves[year_number,1:length+1] = flat_light_curve[mask]
            padding_masks[year_number,1:length+1] = True

        return time_arrays,light_curves,padding_masks,label


    def _get_year_mask_indices(self,time_index):
        return self.year_mask_indices[time_index]

    def _get_years_masks(self,time_index: int):

        mask_indices_array = self._get_year_mask_indices(time_index)

        # Mask for every season
        years_masks = np.zeros((self.years_number,self.time_arrays.shape[1]), dtype=bool)
        for i,(first_index,last_index) in enumerate(mask_indices_array):
            years_masks[i,first_index:last_index+1]=True

        return years_masks

    def __map_over_observations(self,function,dtype=object):
        return np.array([function(time_index) for time_index in range(self.shape[1])],dtype=dtype)

    def seasons_lengths(self,property='time'):

        if str(property).lower() == 'time':

            def getter(time_index):
                return [time[-1]-time[0] for time in self.time_arrays[time_index]]

        elif str(property).lower() == 'pixel':

            def getter(time_index):
                masks_indices = self._get_year_mask_indices(time_index)
                return [idx[-1]-idx[0]+1 for idx in masks_indices]

        else:
            raise ValueError('property should be either "time" or "pixel"')

        return self.__map_over_observations(getter)


class Dataset_Template(Tensor_Dataset):

    def __init__(self, light_curves: np.ndarray, time_arrays: np.ndarray, labels: np.ndarray,
                 simulated_Poisson_scale = 2.5, full_padding_length=165,
                 border_gap_params = (72, 25), border_gap_bounds = (0, np.inf), random_seed=42,
                 light_curve_norm = (1,0),
                 device = None):


        super().__init__()
        self.rng = np.random.default_rng(random_seed)

        border_params = np.array([*border_gap_params,*border_gap_bounds]) / 365

        self.year_dataset = Year_Dataset(light_curves,time_arrays,labels,full_padding_length)
        self.border_gap_sampler = Masking.get_truncnorm_sampler(*border_params,self.rng)

        self.simulated_Poisson_scale = simulated_Poisson_scale
        self.full_padding_length = full_padding_length
        self.normalise_class = Normalise(*light_curve_norm)

        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
        self.device = device

    @property
    def shape(self):
        return self.year_dataset.shape

    def sample_number_of_years(self):

        # pick how many years were observed to make it learn through normalizations
        num_observed_years = self.rng.integers(low=1, high=self.year_dataset.years_number+1)
        if num_observed_years<self.year_dataset.years_number:
            first_year_index = self.rng.integers(low=0, high=self.year_dataset.years_number - num_observed_years)
        else:
            first_year_index = 0

        return num_observed_years,first_year_index

    def get_full_light_curve(self,labels_index, time_index, random_seed_index):
        time_arrays, light_curves, padding_masks, _ = self.year_dataset.get_item(labels_index, time_index, random_seed_index)
        return time_arrays, light_curves, padding_masks

    def subsampling_mask(self,shape,Poisson_scale = 5):
        return Masking.Poisson_process_subsampling_mask(1/self.simulated_Poisson_scale,1/Poisson_scale,shape,self.rng)

    def border_gaps_masks(self,time_arrays):
        start_gap_masks, end_gap_masks = Masking.sample_border_masks(time_arrays,self.border_gap_sampler)
        return start_gap_masks, end_gap_masks



    def augment_time_series(self,time_arrays,value_arrays,padding_masks,
                            labels_index, time_index, random_seed_index,
                            year_index):

        if year_index == 0:
            time_prev = None
            value_prev = None
        else:
            length = padding_masks[year_index - 1, :].sum()
            # the last observed time and value
            time_prev = time_arrays[year_index - 1, length] - 1
            value_prev = value_arrays[year_index - 1, length]

        if year_index == self.year_dataset.years_number - 1:
            # first value from data not big enough to make the full year
            following_index = 1 + self.year_dataset.year_mask_indices[time_index, -1, -1]
            time_next = self.year_dataset.get_time_array(time_index)[
                            following_index] / 365 - self.year_dataset.years_number + 1
            value_next = self.year_dataset.get_light_curve(labels_index, time_index, random_seed_index)[
                following_index]
        else:
            # the first observed time and value
            time_next = time_arrays[year_index + 1, 1] + 1
            value_next = value_arrays[year_index + 1, 1]

        current_series = np.stack([time_arrays[year_index], value_arrays[year_index], padding_masks[year_index]],axis=-1)
        augmented_series, augmentation_mask = self._augment_borders(current_series, time_prev, value_prev, time_next,
                                                              value_next)

        augmented_time_array, augmented_value_array, augmented_padding_mask = np.stack(augmented_series,axis=1)

        return augmented_time_array, augmented_value_array, augmented_padding_mask,augmentation_mask


    def _interpolate_bspline(self, target_time, time_prev, time_next, value_prev, value_next):
        pseudo = (target_time - time_prev) / (time_next - time_prev)
        value = pseudo * value_next + (1 - pseudo) * value_prev
        return value

    def _augment_borders(self, time_series, time_prev = None, value_prev = None, time_next = None, value_next = None):
        """
        Augments borders of time series 'series_cur' using spline interpolated value from previous and next time series
        ----------
        time_series: array (N,3) [time_array,value_array,padding_mask]
            Current time series to be augmented in the start and end. time in [0,1]
        time_prev: float
            time of last measurement before the time series. in [-1,0]
        value_prev: float
            value of last measurement before the time series.
        time_next: float
            time of last measurement before the time series. in [1,2]
        value_next: float
            value of last measurement before the time series.
        -------
        Returns
        augmented_time_series: array (N,3) [time_array,value_array,padding_mask]
            there are two additional values corresponding to time=0 and time=1
        """

        length = time_series[:, 2].sum().astype(int)
        # augmentation in these time points

        # deal with case when we have 0. or 1. in the borders
        time_series[0, 0] = 0.
        time_series[length + 1, 0] = 1.
        augmentation_mask = np.zeros_like(time_series[:,2],dtype=bool)

        if time_series[1,0]!=0.:
            time_series[0, 0] = 0.
            augmentation_mask[0] = True

            # augment in the left edge point t=0
            if (time_prev is None) or (value_prev is None):
                # left edge padding
                time_series[0, 1] = time_series[1, 1]
            else:
                time_series[0, 1] = self._interpolate_bspline(target_time=0, time_prev=time_prev, time_next=time_series[1, 0],
                                                               value_prev=value_prev, value_next=time_series[1, 1])

        if time_series[length,0] != 1.:
            time_series[length + 1, 0] = 1.
            augmentation_mask[length + 1] = True

            # augment in the right edge point t=1
            if (time_next is None) or (value_next is None):
                # right edge padding
                time_series[length + 1, 1] = time_series[length, 1]
            else:
                time_series[length + 1, 1] = self._interpolate_bspline(target_time=1., time_prev=time_series[length, 0], time_next=time_next,
                                                                       value_prev=time_series[length, 1], value_next=value_next)

        time_series[:,2] = np.logical_or(time_series[:,2],augmentation_mask)
        return time_series,augmentation_mask


    def _value_normalization(self,value_array):
        # I also need to rename all "light" to "value"

        value_bounds = (np.max(value_array), np.min(value_array))
        light_normalisation = self.normalise_class.light_curve_normalisation(*value_bounds)

        return light_normalisation

    def _combine_masks(self,padding_masks,subsampling_masks,start_gap_masks,end_gap_masks,is_start_masked = None, is_end_masked = None):

        if is_start_masked is None:
            is_start_masked = np.zeros_like(start_gap_masks,dtype=bool)

        if is_end_masked is None:
            is_end_masked = np.zeros_like(end_gap_masks,dtype=bool)

        return padding_masks & subsampling_masks & \
               ~(start_gap_masks * is_start_masked[:, None]) & ~(end_gap_masks * is_end_masked[:, None])

    def get_item(self, labels_index, time_index, random_seed_index):
        # meant to be redefined in subclasses for training purposes
        # this is an example of full use of functionality

        # Full simulated time series
        full_time_arrays, full_value_arrays, full_padding_masks = self.get_full_light_curve(labels_index, time_index,
                                                                             random_seed_index)

        num_observed_years, first_year_index = self.sample_number_of_years()
        num_year = np.random.randint(low=0, high=num_observed_years)
        year_index = first_year_index + num_year

        time_arrays = full_time_arrays[first_year_index:first_year_index+num_observed_years]
        value_arrays = full_value_arrays[first_year_index:first_year_index+num_observed_years]
        padding_masks = full_padding_masks[first_year_index:first_year_index+num_observed_years]

        # make masks for poisson subsampling and border gaps
        subsampling_masks = self.subsampling_mask(time_arrays.shape)
        start_gap_masks, end_gap_masks = self.border_gaps_masks(time_arrays)
        # whether to mask gaps or not
        is_start_masked, is_end_masked = np.random.randint(0, 2, size=(2,num_observed_years), dtype=bool)
        # half-sampled cadence that should match the observations. No gaps in start and end of the season
        observed_mask = padding_masks & subsampling_masks & \
                        ~(start_gap_masks * is_start_masked[:,None]) & ~(end_gap_masks * is_end_masked[:,None])

        #figure out scaling of light

        # Normalisation as if we observed N seasons and normalise them
        concat_values = value_arrays[observed_mask]
        value_bounds = (np.max(concat_values), np.min(concat_values))
        value_normalisation = self.normalise_class.light_curve_normalisation(*value_bounds)

        # add values on the left and right borders of time series (t=0,t=1)
        # using bspline interpolation with previous and next observed time series
        augmented_time_array, augmented_value_array, augmented_padding_mask, augmentation_mask = \
            self.augment_time_series(full_time_arrays, full_value_arrays, full_padding_masks,
                                     labels_index, time_index,random_seed_index,year_index)

        target_value_array = self.normalise_class.normalise_light_curve(augmented_value_array, *value_normalisation)
        target_value_array *= augmented_padding_mask

        target = np.stack([augmented_time_array, target_value_array, augmented_padding_mask],axis=1)
        context = target * observed_mask[num_year,:,None]

        masks = np.stack([subsampling_masks[num_year], start_gap_masks[num_year], end_gap_masks[num_year], augmentation_mask],axis=1)
        random_params = np.array([num_observed_years, first_year_index, num_year, is_start_masked[num_year], is_end_masked[num_year]])

        labels = self.year_dataset.get_labels(labels_index)
        normalised_labels = self.normalise_class.normalise_label(labels, *value_normalisation, time_factor=1.)
        out_labels = np.array([normalised_labels[0], *np.log10(normalised_labels[1:])])

        return context,target,masks,out_labels,random_params

    def collate_fn(self, batch):
        # meant to be redefined in subclasses for training purposes
        # this is an example of full use of functionality

        context,target,masks, labels, random_params,  indices = np.stack(batch, axis=1)

        labels = np.stack(labels,axis=0)

        return [self._to_tensor(x) for x in [context,target,masks, labels]] + [random_params, indices]

    def _to_tensor(self, array):
        return torch.from_numpy(np.stack(array)).type(torch.float32).to(self.device)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return (*item,index)



