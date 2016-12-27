import itertools
import numpy as np

from ..exception import DataInvalid

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
        print("No lengths or starts supplied...")
        raise
    if starts is None:
        starts = np.append([0],np.cumsum(lengths)[:-1])
    iis_flat = iis_flat[0]
    first_dimension = [
        np.where(starts<=ii)[0][-1] for ii in iis_flat]
    second_dimension = [
        iis_flat[num]-starts[first_dimension[num]] \
        for num in range(len(iis_flat))]
    return (first_dimension,second_dimension)

def _convert_from_2d(iis_ragged, lengths=None, starts=None):
    if lengths is None and starts is None:
        print("No lengths or starts supplied...")
        raise
    if starts is None:
        starts = np.append([0],np.cumsum(lengths)[:-1])
    first_dimension,second_dimension = iis_ragged
    first_dimension = np.array(first_dimension)
    second_dimension = np.array(second_dimension)
    first_dimension_neg_iis = np.where(first_dimension<0)[0]
    second_dimension_neg_iis = np.where(second_dimension<0)[0]
    if len(first_dimension_neg_iis) > 0:
        first_dimension[first_dimension_neg_iis] += len(starts)
    if len(second_dimension_neg_iis) > 0:
        if lengths is None:
            print("Must supply lengths if indices are negative")
            raise
        second_dimension[second_dimension_neg_iis] += lengths[
            first_dimension[second_dimension_neg_iis]]
    if lengths is not None:
        if np.any(lengths[first_dimension] < second_dimension):
            print("indices requested do not exist!..")
            raise
    iis_flat = starts[first_dimension]+second_dimension
    return (iis_flat,)

def _slice_to_list(slice_func, length=None):
    start = slice_func.start
    if start is None:
        start = 0
    elif start < 0:
        if length is None:
            print("Must know length of array to slice negative numbers")
        start = length+start
    stop = slice_func.stop
    if stop is None and length is None:
        print("Must know length of array to slice")
        raise
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
        print(
            "Error: List of length "+len(list_to_partition)+
            " does not equal lengths to partition "+
            str(np.sum(partition_lengths)))
        raise
    partitioned_list = []
    start = 0
    for num in range(len(partition_lengths)):
        stop = start+partition_lengths[num]
        partitioned_list.append(list(list_to_partition[start:stop]))
        start = stop
    return np.array(partitioned_list)

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
    def __init__(self, array):
        self.array_ = np.array(array)
        self.lengths_ = np.array([len(i) for i in array])
        self.starts_ = np.append([0],np.cumsum(self.lengths_)[:-1])
        self.data_ = np.concatenate(array)
    def __repr__(self):
        return self.array_.__repr__()
    def __str__(self):
        return self.array_.__str__()
    def __eq__(self, value):
        return self.data_==value
    def __lt__(self, value):
        return self.data_<value
    def __le__(self, value):
        return self.data_<=value
    def __gt__(self, value):
        return self.data_>value
    def __ge__(self, value):
        return self.data_>=value
    def __ne__(self, value):
        return self.data_!=value
    def __getitem__(self, iis):
        # If the input is a slice or pull in the first dimension, returns a
        # slice or pull of the original array. If the input is a tuple
        # containing a list of indices or one or more slices, the 
        # indices/slices are converted from the 2d-form to the 1d-form
        if (type(iis) is int) or (type(iis) is slice):
            return self.array_[iis]
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
                iis_tmp = np.array(
                    list(
                        itertools.product(
                            first_dimension_iis, second_dimension_iis))).T
                iis = (iis_tmp[0],iis_tmp[1])
                output_unformatted = self.data_[
                    _convert_from_2d(
                        iis, lengths=self.lengths_, starts=self.starts_)]
                return np.array(_chunk(output_unformatted,len(second_dimension_iis)))
            elif type(second_dimension) is slice:
                return self.array_[first_dimension][second_dimension]
            else:
                return self.data_[
                        _convert_from_2d(
                            iis, lengths=self.lengths_, starts=self.starts_)]
    def __setitem__(self, iis, value):
        if (type(iis) is int) or (type(iis) is slice):
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
                iis_tmp = np.array(
                    list(
                        itertools.product(
                            first_dimension_iis, second_dimension_iis))).T
                iis = (iis_tmp[0],iis_tmp[1])
                self.data_[
                    _convert_from_2d(
                        iis, lengths=self.lengths_, starts=self.starts_)] = value
                self.array_ = _partition_list(self.data_,self.lengths_)
            elif type(second_dimension) is slice:
                self.array_[first_dimension][second_dimension] = value
                self.__init__(self.array_)
            else:
                self.data_[
                    _convert_from_2d(
                        iis, lengths=self.lengths_, starts=self.starts_)] = value
                self.array_ = _partition_list(self.data_,self.lengths_)
    def __len__(self):
        return len(self.array_)
    def __add__(self, other):
        if type(other) is type(self):
            other = other.data_
        return self.data_ + other
    def __sub__(self, other):
        if type(other) is type(self):
            other = other.data_
        return self.data_ - other
    def __mul__(self, other):
        if type(other) is type(self):
            other = other.data_
        return self.data_ * other
    def __floordiv__(self, other):
        if type(other) is type(self):
            other = other.data_
        return self.data_ // other
    def __truediv__(self, other):
        if type(other) is type(self):
            other = other.data_
        return self.data_ / other
    def __pow__(self, other):
        if type(other) is type(self):
            other = other.data_
        return self.data_ ** other
    def format(self, values):
        return _partition_list(values,self.lengths_)
    def where(self,mask):
        iis_flat = np.where(mask)
        return _convert_from_1d(iis_flat,starts=self.starts_)
