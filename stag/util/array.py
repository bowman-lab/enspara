import collections
import itertools
import numpy as np

from mdtraj import io
from ..exception import DataInvalid, ImproperlyConfigured

def partition_list(list_to_partition, partition_lengths):
    if np.sum(partition_lengths) != len(list_to_partition):
        raise DataInvalid(
            "List of length {} does not equal lengths to partition {}.".format(
                list_to_partition, partition_lengths))

    partitioned_list = np.full(
        shape=(len(partition_lengths), max(partition_lengths)),
        dtype=list_to_partition.dtype,
        fill_value=-1)

    start = 0
    for num in range(len(partition_lengths)):
        stop = start+partition_lengths[num]
        np.copyto(partitioned_list[num][0:stop-start],
                  list_to_partition[start:stop])
        start = stop

    # this call will mask out all 'invalid' values of partitioned list, in this
    # case all the np.nan values that represent the padding used to make the
    # array square.
    partitioned_list = np.ma.masked_less(partitioned_list, 0, copy=False)

    return partitioned_list


def partition_indices(indices, traj_lengths):
    '''
    Similar to _partition_list in function, this function uses
    `traj_lengths` to determine which 2d trajectory-list index matches
    the given 1d concatenated trajectory index for each index in
    indices.
    '''

    partitioned_indices = []
    for index in indices:
        trj_index = 0
        for traj_len in traj_lengths:
            if traj_len > index:
                partitioned_indices.append((trj_index, index))
                break
            else:
                index -= traj_len
                trj_index += 1

    return partitioned_indices

def _convert_from_1d(iis_flat, lengths=None, starts=None):
    if lengths is None and starts is None:
        raise ImproperlyConfigured(
            'No lengths or starts supplied')
    if starts is None:
        starts = np.append([0],np.cumsum(lengths)[:-1])
    iis_flat = iis_flat[0]
    first_dimension = [
        np.where(starts<=ii)[0][-1] for ii in iis_flat]
    second_dimension = [
        iis_flat[num]-starts[first_dimension[num]] \
        for num in range(len(iis_flat))]
    return (first_dimension,second_dimension)

def _handle_negative_indices(
        first_dimension, second_dimension, lengths=None, starts=None):
    if type(first_dimension) is not np.ndarray:
        first_dimension = np.array(first_dimension)
    if type(second_dimension) is not np.ndarray:
        second_dimension = np.array(second_dimension)
    # remove negative indices from first dimension
    first_dimension_neg_iis = np.where(first_dimension<0)[0]
    second_dimension_neg_iis = np.where(second_dimension<0)[0]
    while len(first_dimension_neg_iis) > 0:
        if first_dimension.size > 1:
            first_dimension[first_dimension_neg_iis] += len(starts)
        else:
            first_dimension += len(starts)
        first_dimension_neg_iis = np.where(first_dimension<0)[0]
    # remove negative indices from second dimension
    while len(second_dimension_neg_iis) > 0:
        if lengths is None:
            raise ImproperlyConfigured(
                'Must supply lengths if indices are negative.')
        if second_dimension.size > 1:
            if first_dimension.size > 1:
                second_dimension[second_dimension_neg_iis] += lengths[
                    first_dimension[second_dimension_neg_iis]]
            else:
                second_dimension[second_dimension_neg_iis] += lengths[
                    first_dimension]
        else:
            second_dimension += lengths[first_dimension]
        second_dimension_neg_iis = np.where(second_dimension<0)[0]
    return first_dimension, second_dimension

def _convert_from_2d(iis_ragged, lengths=None, starts=None, error_check=True):
    if lengths is None and starts is None:
        raise ImproperlyConfigured(
            'No lengths or starts supplied')
    if starts is None:
        starts = np.append([0],np.cumsum(lengths)[:-1])
    first_dimension,second_dimension = iis_ragged
    first_dimension = np.array(first_dimension)
    second_dimension = np.array(second_dimension)
    # Account for iis = ([0,1,2],4)
    if first_dimension.size > 1 and second_dimension.size == 1:
        second_dimension = np.array(
            [second_dimension for n in first_dimension])
    first_dimension, second_dimension = _handle_negative_indices(
        first_dimension, second_dimension, lengths=lengths, starts=starts)
    # Check for index error
    if lengths is not None and error_check:
        if np.any(lengths[first_dimension] <= second_dimension):
            raise IndexError
    iis_flat = starts[first_dimension]+second_dimension
    return (iis_flat,)

def _slice_to_list(slice_func, length=None):
    start = slice_func.start
    if start is None:
        start = 0
    elif start < 0:
        if length is None:
            raise ImproperlyConfigured(
                'Must supply length of array if slicing to negative indices')
        start = length+start
    stop = slice_func.stop
    if stop is None and length is None:
        raise ImproperlyConfigured(
            'Must supply length of array if stop is None')
    if stop is None:
        stop = length
    elif stop < 0:
        stop = length+stop
    step = slice_func.step
    if step is None:
        step = 1
    elif step < 0 and stop is None and start is None:
        start = copy.copy(stop)
        stop = -1
    return range(start,stop,step)

def _chunk(l, n):
    """Yield successive n-sized chunks from l."""
    chunk_list = []
    for i in range(0, len(l), n):
        chunk_list.append(l[i:i + n])
    return chunk_list

def _partition_list(list_to_partition, partition_lengths):
    if np.sum(partition_lengths) != len(list_to_partition):
        raise DataInvalid(
            'Number of elements in list (%d) does not equal' % len(list_to_partition)+\
            ' the sum of the lengths to partition (%d)' % np.sum(partition_lengths))
    partitioned_list = []
    start = 0
    for num in range(len(partition_lengths)):
        stop = start+partition_lengths[num]
        partitioned_list.append(list_to_partition[start:stop])
        start = stop
    return partitioned_list

def _flatten(l):
    """Flattens an array of various sized elements"""
    for element in l:
        if isinstance(element, collections.Iterable) and not \
                isinstance(element, (str, bytes)):
            yield from _flatten(element)
        else:
            yield element

def _is_iterable(iterable):
    iterable_bool = isinstance(iterable,collections.Iterable) and not \
        isinstance(iterable, (str,bytes))
    return iterable_bool

def _ensure_ragged_data(array):
    if not _is_iterable(array):
        raise DataInvalid('Must supply an array or list of arrays as input')
    if len(array) == 0:
        pass
    if len(array) == 1:
        pass
    else:
        for num in range(len(array)-1):
            if _is_iterable(array[num]) != _is_iterable(array[num+1]):
                raise DataInvalid(
                    'The array elements in the input are not consistent.')
    return

def _remove_outliers(iis, lengths):
    first_dimension,second_dimension = iis
    if type(first_dimension) is not np.ndarray:
        first_dimension = np.array(first_dimension)
    if type(second_dimension) is not np.ndarray:
        second_dimension = np.array(second_dimension)
    unique_firsts = np.unique(first_dimension)
    new_lengths = np.array([],dtype=int)
    for num in range(len(unique_firsts)):
        max_len = lengths[unique_firsts[num]]
        first_dimension_iis = np.where(first_dimension==unique_firsts[num])
        iis_to_nix = first_dimension_iis[0][
            np.where(second_dimension[first_dimension_iis] >= max_len)]
        first_dimension = np.delete(first_dimension,iis_to_nix)
        second_dimension = np.delete(second_dimension,iis_to_nix)
        new_length = len(first_dimension_iis[0])-len(iis_to_nix)
        new_lengths = np.append(new_lengths,new_length)
    sort_iis = np.lexsort((second_dimension,first_dimension))
    return first_dimension[sort_iis],second_dimension[sort_iis],new_lengths
        
class ragged_array(object):
    """ragged_array class
    
    The ragged array class takes an array of arrays with various lengths and
    returns an object that allows for indexing, slicing, and querying as if a
    2d array. The array is concatenated and stored as a 1d array.

    Attributes
    ----------
    array_ : array, [n,]
        The original input array.
    data_ : array, 
        The concatenated array.
    lengths_ : array, [n]
        The length of each sub-array within array_
    starts_ : array, [n]
        The indices of the 1d array that correspond to the first element in
        array_.
    """
    def __init__(self, array, lengths=None, error_checking=True):
        # TODO:
        # 1) update tests 
        if error_checking is True: 
            array = np.array(list(array))
            if len(array) > 20000:
                print(
                    "WARNING: error checking is turned off for ragged arrays "+\
                    "with first dimension greater than 20000")
            else:
                _ensure_ragged_data(array)
        # concatenate data if list of lists
        if len(array) > 0:
            if _is_iterable(array[0]):
                self.data_ = np.concatenate(array)
            else:
                self.data_ = np.array(array)
        # new array greater with >0 elements
        if (lengths is None) and (len(array) > 0):
            # array of arrays
            if _is_iterable(array[0]):
                self.lengths_ = np.array([len(i) for i in array],dtype=int)
                self.array_ = np.array(
                    _partition_list(self.data_, self.lengths_), dtype='O')
            # array of single values
            else:
                self.lengths_ = np.array([len(array)],dtype=int)
                self.array_ = self.data_.reshape((1,self.lengths_[0]))
        # null array
        elif lengths is None:
            self.lengths_ = np.array([],dtype=int)
            self.array_ = []
        # rebuild array from 1d and lengths
        else:
            self.array_ = np.array(
                _partition_list(self.data_, lengths), dtype='O')
            self.lengths_ = np.array(lengths)
        self.starts_ = np.append([0],np.cumsum(self.lengths_)[:-1])
    # Built in functions
    def __len__(self):
        return len(self.array_)
    def __repr__(self):
        return self.array_.__repr__()
    def __str__(self):
        return self.array_.__str__()
    def __getitem__(self, iis):
        if type(iis) is int: 
            return self.array_[iis]
        elif (type(iis) is slice) or (type(iis) is list) \
                or (type(iis) is np.ndarray):
            return ragged_array(self.array_[iis])
        elif type(iis) is tuple:
            first_dimension,second_dimension = iis
            if type(first_dimension) is slice:
                first_dimension_iis = _slice_to_list(
                    first_dimension, length=len(self.lengths_))
                if type(second_dimension) is slice:
                    second_dimension_length = \
                        self.lengths_[first_dimension_iis].max()
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
                elif type(second_dimension) is int:
                    second_dimension_iis = [second_dimension]
                else:
                    second_dimension_iis = second_dimension
            elif type(second_dimension) is slice:
                if type(first_dimension) is int:
                    return self.array_[first_dimension][second_dimension]
                else:
                    first_dimension_iis = first_dimension
                    second_dimension_length = \
                        self.lengths_[first_dimension_iis].max()
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
            else:
                return self.data_[
                        _convert_from_2d(
                            iis, lengths=self.lengths_, starts=self.starts_)]
            iis_tmp = np.array(
                list(
                    itertools.product(
                        first_dimension_iis, second_dimension_iis))).T
            # If indices do not exist, remove them
            new_first_dimension, new_second_dimension, new_lengths = \
                _remove_outliers(iis_tmp, self.lengths_)
            iis = (new_first_dimension, new_second_dimension)
            output_unformatted = self.data_[
                _convert_from_2d(
                    iis, lengths=self.lengths_,
                    starts=self.starts_)]
            return ragged_array(output_unformatted, lengths=new_lengths)
        elif type(iis) is type(self):
            iis = ragged_array.where(iis)
            return self.__getitem__(iis)
    def __setitem__(self, iis, value):
        if (type(iis) is int) or (type(iis) is slice) or \
                (type(iis) is list) or (type(iis) is np.ndarray):
            self.array_[iis] = value
            self.__init__(self.array_)
        elif type(iis) == tuple:
            first_dimension,second_dimension = iis
            if type(first_dimension) is slice:
                first_dimension_iis = _slice_to_list(
                    first_dimension, length=len(self.lengths_))
                if type(second_dimension) is slice:
                    second_dimension_length = \
                        self.lengths_[first_dimension_iis].min()
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
                elif type(second_dimension) is int:
                    second_dimension_iis = [second_dimension]
                else:
                    second_dimension_iis = second_dimension
            elif type(second_dimension) is slice:
                if type(first_dimension) is int:
                    self.array_[first_dimension][second_dimension] = value
                    self.__init__(self.array_)
                    return
                else:
                    first_dimension_iis = first_dimension
                    second_dimension_length = \
                        self.lengths_[first_dimension_iis].min()
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
            else:
                iis_1d = _convert_from_2d(
                    iis, lengths=self.lengths_, starts=self.starts_)
                if _is_iterable(value):
                    if _is_iterable(value[0]):
                        value_1d = np.concatenate(value)
                    else:
                        value_1d = value
                else:
                    value_1d = value
                self.data_[iis_1d] = value_1d
                self.array_ = np.array(
                    _partition_list(self.data_,self.lengths_), dtype='O')
                return
            iis_tmp = np.array(
                list(
                    itertools.product(
                        first_dimension_iis, second_dimension_iis))).T
            # If indices do not exist, remove them
            new_first_dimension, new_second_dimension, new_lengths = \
                _remove_outliers(iis_tmp, self.lengths_)
            iis = (new_first_dimension, new_second_dimension)
            iis_1d = _convert_from_2d(
                iis,lengths=self.lengths_,starts=self.starts_)
            if _is_iterable(value):
                value_1d = np.concatenate(value)
            else:
                value_1d = value
            self.data_[iis_1d] = value_1d
            self.array_ = np.array(
                _partition_list(self.data_,self.lengths_),dtype='O')
        elif type(iis) is type(self):
            iis = ragged_array.where(iis)
            self.__setitem__(iis,value)
    def __eq__(self, other):
        return self.map_operator('__eq__', other)
    def __lt__(self, other):
        return self.map_operator('__lt__', other)
    def __le__(self, other):
        return self.map_operator('__le__', other)
    def __gt__(self, other):
        return self.map_operator('__gt__', other)
    def __ge__(self, other):
        return self.map_operator('__ge__', other)
    def __ne__(self, other):
        return self.map_operator('__ne__', other)
    def __add__(self, other):
        return self.map_operator('__add__', other)
    def __radd__(self, other):
        return self.map_operator('__radd__', other)
    def __sub__(self, other):
        return self.map_operator('__sub__', other)
    def __rsub__(self, other):
        return self.map_operator('__rsub__', other)
    def __mul__(self, other):
        return self.map_operator('__mul__', other)
    def __rmul__(self, other):
        return self.map_operator('__rmul__', other)
    def __truediv__(self, other):
        return self.map_operator('__truediv__', other)
    def __rtruediv__(self, other):
        return self.map_operator('__rtruediv__', other)
    def __floordiv__(self, other):
        return self.map_operator('__floordiv__', other)
    def __rfloordiv__(self, other):
        return self.map_operator('__rfloordiv__', other)
    def __pow__(self, other):
        return self.map_operator('__pow__', other)
    def __rpow__(self, other):
        return self.map_operator('__rpow__', other)
    def map_operator(self, operator, other):
        if type(other) is type(self):
            other = other.data_
        new_data = getattr(self.data_, operator)(other)
        return ragged_array(
            array=new_data, lengths=self.lengths_, error_checking=False)
    # Non-built in functions
    def all(self):
        return np.all(self.data_)
    def any(self):
        return np.any(self.data_)
    def where(mask):
        iis_flat = np.where(mask.data_)
        return _convert_from_1d(iis_flat,starts=mask.starts_)
    def append(self,values):
        if type(values) is type(self):
            values = values.array_
        if len(self.data_) == 0:
            self.__init__(values)
        else:
            concat_values = list(_flatten(values))
            self.data_ = np.append(self.data_, concat_values)
            if _is_iterable(values):
                if _is_iterable(values[0]):
                    new_lengths = np.array([len(i) for i in values])
                else:
                    new_lengths = [len(values)]
            else:
                raise DataInvalid(
                    'Expected an array of values or a ragged array')
            self.lengths_ = np.append(self.lengths_, new_lengths)
            self.array_ = np.array(
                _partition_list(self.data_, self.lengths_), dtype='O')
            self.starts_ = np.append([0],np.cumsum(self.lengths_)[:-1])
    def flatten(self):
        return np.array(list(_flatten(self.data_)))
    def save(self,output_name):
        to_save = {'array': self.data_, 'lengths': self.lengths_}
        io.saveh(output_name,**to_save)
    def load(input_name):
        ragged_load = io.loadh(input_name)
        return ragged_array(
            ragged_load['array'],lengths=ragged_load['lengths'])
