import logging
import collections
import itertools
import numpy as np
import copy

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
    """Given 1d indices, converts to 2d."""
    if lengths is None and starts is None:
        raise ImproperlyConfigured(
            'No lengths or starts supplied')
    if starts is None:
        starts = np.append([0], np.cumsum(lengths)[:-1])
    iis_flat = iis_flat[0]
    first_dimension = [
        np.where(starts <= ii)[0][-1] for ii in iis_flat]
    second_dimension = [
        iis_flat[num]-starts[first_dimension[num]]
        for num in range(len(iis_flat))]
    return (first_dimension, second_dimension)


def _handle_negative_indices(
        first_dimension, second_dimension, lengths=None, starts=None):
    """Given 2d indices as first_dimenion and second_dimension, converts
       any negative index to a positive one."""
    if type(first_dimension) is not np.ndarray:
        first_dimension = np.array(first_dimension)
    if type(second_dimension) is not np.ndarray:
        second_dimension = np.array(second_dimension)
    # remove negative indices from first dimension
    first_dimension_neg_iis = np.where(first_dimension < 0)[0]
    second_dimension_neg_iis = np.where(second_dimension < 0)[0]
    while len(first_dimension_neg_iis) > 0:
        if first_dimension.size > 1:
            first_dimension[first_dimension_neg_iis] += len(starts)
        else:
            first_dimension += len(starts)
        first_dimension_neg_iis = np.where(first_dimension < 0)[0]
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
        second_dimension_neg_iis = np.where(second_dimension < 0)[0]
    return first_dimension, second_dimension


def _convert_from_2d(iis_ragged, lengths=None, starts=None, error_check=True):
    """Given indices in 2d, returns the corresponding 1d indices.
       Requires either lengths or starts."""
    if lengths is None and starts is None:
        raise ImproperlyConfigured(
            'No lengths or starts supplied')
    if starts is None:
        starts = np.append([0], np.cumsum(lengths)[:-1])
    first_dimension, second_dimension = iis_ragged
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
    """Converts a slice to a list. Requires the length of the array if
       slicing to a negative index or there is no stopping criterion."""
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
    return range(start, stop, step)


def _partition_list(list_to_partition, partition_lengths):
    """Partitions list by partition lengths. Different from previous
       versions in that is does not return a masked array."""
    if np.sum(partition_lengths) != len(list_to_partition):
        raise DataInvalid(
            'Number of elements in list (%d) does not equal' %
            len(list_to_partition) +
            ' the sum of the lengths to partition (%d)' %
            np.sum(partition_lengths))
    partitioned_list = []
    start = 0
    for num in range(len(partition_lengths)):
        stop = start+partition_lengths[num]
        partitioned_list.append(list_to_partition[start:stop])
        start = stop
    return partitioned_list


def _is_iterable(iterable):
    """Indicates if the input is iterable but not due to being a string or
       bytes. Returns a boolean value."""
    iterable_bool = isinstance(iterable, collections.Iterable) and not \
        isinstance(iterable, (str, bytes))
    return iterable_bool


def _ensure_ragged_data(array):
    """Raises an exception if the input is either:
       1) not an array of arrays or 2) not a 1 dimensional array"""
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
    """This is a helper function for indexing on the RaggedArray. If indices
       are non-existant, this function removes them with knowledge of the
       lengths of each ragged array. Returns the valid indices and the
       lengths to partition them by."""
    # make the first and second dimensions numpy arrays
    first_dimension, second_dimension = iis
    if type(first_dimension) is not np.ndarray:
        first_dimension = np.array(first_dimension)
    if type(second_dimension) is not np.ndarray:
        second_dimension = np.array(second_dimension)
    # get the unique first
    unique_firsts = np.unique(first_dimension)
    new_lengths = np.array([], dtype=int)
    # Iterate over all of the unique first dimension indices
    # and remove outliers
    for num in range(len(unique_firsts)):
        max_len = lengths[unique_firsts[num]]
        first_dimension_iis = np.where(first_dimension == unique_firsts[num])
        iis_to_nix = first_dimension_iis[0][
            np.where(second_dimension[first_dimension_iis] >= max_len)]
        first_dimension = np.delete(first_dimension, iis_to_nix)
        second_dimension = np.delete(second_dimension, iis_to_nix)
        new_length = len(first_dimension_iis[0])-len(iis_to_nix)
        new_lengths = np.append(new_lengths, new_length)
    sort_iis = np.lexsort((second_dimension, first_dimension))
    return first_dimension[sort_iis], second_dimension[sort_iis], new_lengths


def _format__arrayline(_arrayline, operator):
    """Formats a single line of an array"""
    formatted = getattr(_arrayline, operator)().split(')')[0].split('(')[-1]
    return formatted


def _format_array(array, operator):
    """Formats a ragged array output"""
    # Determine the correct formatting for the operator
    if operator == '__repr__':
        header = 'RaggedArray([\n'
        aftermath = '])'
        line_spacing = '    '
    elif operator == '__str__':
        header = '['
        aftermath = ']'
        line_spacing = ''
    body = []
    # If the length of the array is greater than 6, generates an elipses
    if len(array) > 6:
        for i in [0, 1, 2]:
            body.append(
                line_spacing+_format__arrayline(array[i], operator))
        body.append(line_spacing+'...')
        for i in [-3, -2, -1]:
            body.append(
                line_spacing+_format__arrayline(array[i], operator))
        return "".join([header, ",\n".join(body), aftermath])
    else:
        for i in range(len(array)):
            body.append(
                line_spacing+_format__arrayline(array[i], operator))
        return "".join([header, ",\n".join(body), aftermath])


class RaggedArray(object):
    """RaggedArray class

    The RaggedArray class takes an array of arrays with various lengths and
    returns an object that allows for indexing, slicing, and querying as if a
    2d array. The array is concatenated and stored as a 1d array.

    Attributes
    ----------
    _array : array, [n,]
        The original input array.
    _data : array,
        The concatenated array.
    lengths : array, [n]
        The length of each sub-array within _array
    _starts : array, [n]
        The indices of the 1d array that correspond to the first element in
        _array.
    """

    __slots__ = ('_data', '_array', 'lengths', '_starts')

    def __init__(self, array, lengths=None, error_checking=True):
        # Check that input is proper (array of arrays)
        if error_checking is True:
            array = np.array(list(array))
            if len(array) > 20000:
                logging.warning(
                    "error checking is turned off for ragged arrays "
                    "with first dimension greater than 20000")
            else:
                _ensure_ragged_data(array)
        # concatenate data if list of lists
        if len(array) > 0:
            if _is_iterable(array[0]):
                self._data = np.concatenate(array)
            else:
                self._data = np.array(array)
        # new array greater with >0 elements
        if (lengths is None) and (len(array) > 0):
            # array of arrays
            if _is_iterable(array[0]):
                self.lengths = np.array([len(i) for i in array], dtype=int)
                self._array = np.array(
                    _partition_list(self._data, self.lengths), dtype='O')
            # array of single values
            else:
                self.lengths = np.array([len(array)], dtype=int)
                self._array = self._data.reshape((1, self.lengths[0]))
        # null array
        elif lengths is None:
            self.lengths = np.array([], dtype=int)
            self._array = []
        # rebuild array from 1d and lengths
        else:
            self._array = np.array(
                _partition_list(self._data, lengths), dtype='O')
            self.lengths = np.array(lengths)
        self._starts = np.append([0], np.cumsum(self.lengths)[:-1])

    # Built in functions
    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return _format_array(self._array, '__repr__')
    def __str__(self):
        return _format_array(self._array, '__str__')

    def __getitem__(self, iis):
        # ints are handled by numpy
        if type(iis) is int:
            return self._array[iis]
        # slices and lists are handled by numpy, but return a RaggedArray
        elif (type(iis) is slice) or (type(iis) is list) \
                or (type(iis) is np.ndarray):
            return RaggedArray(self._array[iis])
        # tuples get index conversion from 2d to 1d
        elif type(iis) is tuple:
            first_dimension, second_dimension = iis
            # if the first dimension is a slice, converts both sets of indices
            if type(first_dimension) is slice:
                first_dimension_iis = _slice_to_list(
                    first_dimension, length=len(self.lengths))
                # if the second dimension is a slice, pick the maximum length
                # of all arrays for conversion of slice to list. Indices that
                # do not exist are later removed.
                if type(second_dimension) is slice:
                    second_dimension_length = \
                        self.lengths[first_dimension_iis].max()
                    if second_dimension.stop is not None:
                        if second_dimension.stop > second_dimension_length:
                            raise IndexError
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
                # make sure the second dimension is a list
                elif type(second_dimension) is int:
                    second_dimension_iis = [second_dimension]
                else:
                    second_dimension_iis = second_dimension
            elif type(second_dimension) is slice:
                # if the first dimension is an int, but the second is
                # a slice, numpy can handle it.
                if type(first_dimension) is int:
                    return self._array[first_dimension][second_dimension]
                # if the second dimension is a slice, pick the maximum length
                # of all arrays for conversion of slice to list. Indices that
                # do not exist are later removed.
                else:
                    first_dimension_iis = first_dimension
                    second_dimension_length = \
                        self.lengths[first_dimension_iis].max()
                    if second_dimension.stop is not None:
                        if second_dimension.stop > second_dimension_length:
                            raise IndexError
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
            # If the indices are a tuple, but does not contain a slice,
            # does regular conversion.
            else:
                return self._data[
                        _convert_from_2d(
                            iis, lengths=self.lengths, starts=self._starts)]
            # combinatorically arrange all possible pairs of the
            # different dimensions
            iis_tmp = np.array(
                list(
                    itertools.product(
                        first_dimension_iis, second_dimension_iis))).T
            # If indices do not exist, remove them
            new_first_dimension, new_second_dimension, new_lengths = \
                _remove_outliers(iis_tmp, self.lengths)
            iis = (new_first_dimension, new_second_dimension)
            # format the slices by the correct lengths and return a new
            # RaggedArray object
            output_unformatted = self._data[
                _convert_from_2d(
                    iis, lengths=self.lengths,
                    starts=self._starts)]
            return RaggedArray(output_unformatted, lengths=new_lengths)
        # if the indices are of self, assumes a boolean matrix. Converts
        # bool to indices and recalls __getitem__
        elif type(iis) is type(self):
            iis = RaggedArray.where(iis)
            return self.__getitem__(iis)

    def __setitem__(self, iis, value):
        # ints, slices, lists, and numpy objects are handled by numpy
        if (type(iis) is int) or (type(iis) is slice) or \
                (type(iis) is list) or (type(iis) is np.ndarray):
            self._array[iis] = value
            self.__init__(self._array)
        # tuples get index conversion from 2d to 1d
        elif type(iis) == tuple:
            first_dimension, second_dimension = iis
            # if the first dimension is a slice, converts both sets of indices
            if type(first_dimension) is slice:
                first_dimension_iis = _slice_to_list(
                    first_dimension, length=len(self.lengths))
                # if the second dimension is a slice, pick the maximum length
                # of all arrays for conversion of slice to list. Indices that
                # do not exist are later removed.
                if type(second_dimension) is slice:
                    second_dimension_length = \
                        self.lengths[first_dimension_iis].max()
                    if second_dimension.stop is not None:
                        if second_dimension.stop > second_dimension_length:
                            raise IndexError
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
                # make sure the second dimension is a list
                elif type(second_dimension) is int:
                    second_dimension_iis = [second_dimension]
                else:
                    second_dimension_iis = second_dimension
            elif type(second_dimension) is slice:
                # if the first dimension is an int, but the second is
                # a slice, numpy can handle it.
                if type(first_dimension) is int:
                    self._array[first_dimension][second_dimension] = value
                    self.__init__(self._array)
                    return
                # if the second dimension is a slice, pick the maximum length
                # of all arrays for conversion of slice to list. Indices that
                # do not exist are later removed.
                else:
                    first_dimension_iis = first_dimension
                    second_dimension_length = \
                        self.lengths[first_dimension_iis].max()
                    if second_dimension.stop is not None:
                        if second_dimension.stop > second_dimension_length:
                            raise IndexError
                    second_dimension_iis = _slice_to_list(
                        second_dimension, length=second_dimension_length)
            # If the indices are a tuple, but does not contain a slice,
            # does regular conversion.
            else:
                iis_1d = _convert_from_2d(
                    iis, lengths=self.lengths, starts=self._starts)
                # concatenates values if necessary
                if _is_iterable(value):
                    if _is_iterable(value[0]):
                        value_1d = np.concatenate(value)
                    else:
                        value_1d = value
                else:
                    value_1d = value
                self._data[iis_1d] = value_1d
                self._array = np.array(
                    _partition_list(self._data, self.lengths), dtype='O')
                return
            # combinatorically arrange all possible pairs of the
            # different dimensions
            iis_tmp = np.array(
                list(
                    itertools.product(
                        first_dimension_iis, second_dimension_iis))).T
            # If indices do not exist, remove them
            new_first_dimension, new_second_dimension, new_lengths = \
                _remove_outliers(iis_tmp, self.lengths)
            iis = (new_first_dimension, new_second_dimension)
            iis_1d = _convert_from_2d(
                iis, lengths=self.lengths, starts=self._starts)
            if _is_iterable(value):
                value_1d = np.concatenate(value)
            else:
                value_1d = value
            self._data[iis_1d] = value_1d
            self._array = np.array(
                _partition_list(self._data, self.lengths), dtype='O')
        # if the indices are of self, assumes a boolean matrix. Converts
        # bool to indices and recalls __getitem__
        elif type(iis) is type(self):
            iis = RaggedArray.where(iis)
            self.__setitem__(iis, value)

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
    def __mod__(self, other):
        return self.map_operator('__mod__', other)
    def __rmod__(self, other):
        return self.map_operator('__rmod__', other)
    def map_operator(self, operator, other):
        if type(other) is type(self):
            other = other._data
        new_data = getattr(self._data, operator)(other)
        return RaggedArray(
            array=new_data, lengths=self.lengths, error_checking=False)

    # Non-built in functions
    def all(self):
        return np.all(self._data)

    def any(self):
        return np.any(self._data)

    def where(mask):
        iis_flat = np.where(mask._data)
        return _convert_from_1d(iis_flat, starts=mask._starts)

    def append(self, values):
        # if the incoming values is a RaggedArray, pull just the array
        if type(values) is type(self):
            values = values._array
        # if the current RaggedArray is blank, generate a new one
        # with the values input
        if len(self._data) == 0:
            self.__init__(values)
        else:
            concat_values = np.concatenate(values)
            self._data = np.append(self._data, concat_values)
            # if the values are a list of arrays, add them each individually
            if _is_iterable(values):
                if _is_iterable(values[0]):
                    new_lengths = np.array([len(i) for i in values])
                else:
                    new_lengths = [len(values)]
            else:
                raise DataInvalid(
                    'Expected an array of values or a ragged array')
            # update variables
            self.lengths = np.append(self.lengths, new_lengths)
            self._array = np.array(
                _partition_list(self._data, self.lengths), dtype='O')
            self._starts = np.append([0], np.cumsum(self.lengths)[:-1])

    def flatten(self):
        return self._data.flatten()

    def save(self, output_name):
        to_save = {'array': self._data, 'lengths': self.lengths}
        io.saveh(output_name, **to_save)

    def load(input_name):
        ragged_load = io.loadh(input_name)
        return RaggedArray(
            ragged_load['array'], lengths=ragged_load['lengths'])
