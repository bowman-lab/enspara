import collections
import copy
import itertools
import logging
import numbers
import numpy as np
import resource
import tables
import time
import warnings

from mdtraj import io
from ..exception import DataInvalid, ImproperlyConfigured

logger = logging.getLogger(__name__)


def zeros_like(array, *args, **kwargs):

    if hasattr(array, '_data'):
        flat_arr = np.zeros_like(array._data)
        return RaggedArray(array=flat_arr, lengths=array.lengths)
    else:
        return np.zeros_like(array)


def where(mask):
    """As np.where, but on _either_ RaggedArrays or a numpy array.

    Parameters
    ----------
    mask : array or RaggedArray

    Returns
    -------
    (rows, columns) : (array, array))
    """
    try:
        iis_flat = np.where(mask._data)
        return _convert_from_1d(iis_flat, starts=mask.starts)
    except AttributeError:
        return np.where(mask)


def save(filename, array, compression_level=1, tag='arr'):
    """Save a RaggedArray or numpy ndarray to disk as an HDF5 file.
     Parameters
    ----------
    filename : str
        Path of file to write out (per tables.open_file).
    array : np.ndarray, RaggedArray
        Array to write to disk.
    compression_level : int
        Level of compression to use, 0-9, with 0 meaning no compression.
        Per the pytables Filters complevel flag.
    tag : str, default='array'
        The name under which each row in the ragged array will be saved,
        for example 'array_00'.
    """

    try:
        n_zeros = len(str(len(array.lengths))) + 1
    except AttributeError:
        n_zeros = 1
        array = [array]

    compression = tables.Filters(
        complevel=compression_level,
        complib='zlib',
        shuffle=True)

    with tables.open_file(filename, 'w') as handle:
        for i in range(len(array)):
            subarr = array[i]

            if hasattr(array, '_data'):
                atom = tables.Atom.from_dtype(array._data.dtype)
            else:
                atom = tables.Atom.from_dtype(subarr.dtype)

            t = tag + '_' + str(i).zfill(n_zeros)

            node = handle.create_carray(
                where='/', name=t, atom=atom,
                shape=subarr.shape, filters=compression)

            node[:] = subarr

    return filename


def _save_old_style(output_name, ragged_array):
    """Depricated en bloc RaggedArray saving routine.

    Parameters
    ----------
    output_name : str
        Path of file to write out.
    ragged_array : np.ndarray, RaggedArray
        Array to write to disk.

    See Also
    --------
    mdtraj.io.saveh
    """

    try:
        io.saveh(
            output_name,
            array=ragged_array._data,
            lengths=ragged_array.lengths)
    except AttributeError:
        # A TypeError results when the input is actually an ndarray
        io.saveh(output_name, ragged_array)


def load(input_name, keys=..., stride=1):
    """Load a RaggedArray from the disk. If only 'arr_0' is present in
    the target file, a numpy array is loaded instead.

    Parameters
    ----------
    input_name: filename or file handle
        File from which data will be loaded.
    keys : list, default=...
        If this option is specified, the ragged array is built from this
        list of keys, each of which are assumed to be a row of the final
        ragged array. An ellipsis can be provided to indicate all keys.
    stride: int, default=1
        This option specifies a stride in the second dimension of the
        loaded ragged array. This is equivalent to slicing out
        [:, ::stride], except that it does not load the entire dataset
        into memory.

    Returns
    -------
    ra : RaggedArray
        A ragged array from disk.
    """

    with tables.open_file(input_name) as handle:
        if keys is None:
            if '/lengths' in handle:
                a = RaggedArray(
                    handle.get_node('/array'),
                    lengths=handle.get_node('/lengths'))
                return a[::stride]
            else:
                return handle.get_node('/arr_0')[::stride]
        else:
            if keys is Ellipsis:
                keys = [k.name for k in handle.list_nodes('/')]
            if '/lengths' in handle and '/array' in handle:
                warnings.warn(DeprecationWarning,
                              "Found keys '/lengths' and '/array' in h5 "
                              "file %s, are you sure this isn't an "
                              "old-style h5?", input_name)
            if len(keys) == 1:
                logger.debug("Found only one key ('%s') returning that as "
                             "numpy array", keys[0])
                return handle.get_node('/' + keys[0])[:]

            logger.debug('Loading keys %s into RA', keys)

            shapes = [handle.get_node(where='/', name=k).shape
                      for k in keys]

            if not all(len(shapes[0]) == len(shape) for shape in shapes):
                raise DataInvalid(
                    "Loading a RaggedArray using HDF5 file keys requires "
                    "that all input arrays have the same dimension. Got "
                    "shapes: %s" % shapes)
            for dim in range(1, len(shapes[0])):
                if not all(shapes[0][dim] == shape[dim] for shape in shapes):
                    raise DataInvalid(
                        "Loading a RaggedArray using HDF5 file keys requires "
                        "that all input arrays share nonragged dimensions. "
                        " Dimension  %s didn't match. Got shapes: %s"
                        % (dim, shapes))

            lengths = [(shape[0] + stride - 1) // stride for shape in shapes]
            concat_shape = (sum(lengths),) + (shapes[0][1:])

            dtype = handle.get_node(where='/', name=keys[0]).dtype
            if not all([dtype == handle.get_node(where='/', name=k).dtype
                        for k in keys]):
                raise DataInvalid(
                    "Can't load keys in %s because the keys didn't have all "
                    "the same dtype. Keys were: %s" % (dtype, keys))

            logger.debug('Allocating array of shape %s.', concat_shape)
            tick = time.perf_counter()
            concat = np.zeros(concat_shape, dtype=dtype)
            tock = time.perf_counter()
            logger.debug('Allocated %.3f MB in %.2f min.',
                         concat.data.nbytes / 1024**2, tock - tick)

            logger.debug(
                'Filling array with %s blocks with initial memory '
                'footprint of %.3f GB',
                len(keys),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)

            tick = time.perf_counter()
            start = 0
            for key in keys:
                node = handle.get_node(where='/', name=key)[::stride]
                end = start + len(node)
                concat[start:end] = node
                start = end

            tock = time.perf_counter()
            logger.debug(
                'Filled RaggedArray in %.3f min with %.3f GB memory overhead.',
                (tock - tick) / 60,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
            tick = time.perf_counter()

            handle.close()
            return RaggedArray(array=concat, lengths=lengths, copy=False)


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
    return (np.array(first_dimension), np.array(second_dimension))


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
    if len(first_dimension_neg_iis) > 0:
        if first_dimension.size > 1:
            first_dimension[first_dimension_neg_iis] += len(starts)
        else:
            first_dimension += len(starts)
        if len(np.where(first_dimension < 0)[0]) > 0:
            # TODO: have clear error message here
            raise IndexError()
    # remove negative indices from second dimension
    if len(second_dimension_neg_iis) > 0:
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
        if len(np.where(second_dimension < 0)[0]) > 0:
            # TODO: have clear error message here
            raise IndexError()
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


def partition_list(list_to_partition, partition_lengths):
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
    iterable_bool = isinstance(iterable, collections.abc.Iterable) and not \
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
        line_spacing = '      '
    elif operator == '__str__':
        header = '['
        aftermath = ']'
        line_spacing = ' '
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


def _get_iis_from_slices(first_dimension_iis, second_dimension, lengths):
    """Given the indices of the first dimension, the second dimension
    (as a slice), and the lengths of the ragged dimension, returns the
    2D indices and the new lengths in the ragged dimension."""
    start = second_dimension.start
    stop = second_dimension.stop
    step = second_dimension.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    # handle negative slicing
    if stop is None:
        stops = lengths
    elif stop < 0:
        stops = lengths + stop
    else:
        stops = np.zeros(lengths.shape, dtype=int) + stop
    # if indices go past length, make it go upto length
    iis_to_flat = np.where(stops > lengths)
    stops[iis_to_flat] = lengths[iis_to_flat]
    iis_2d = np.array(
        [np.arange(start, stops[num], step) for num in first_dimension_iis])
    iis_2d_lengths = np.array([len(i) for i in iis_2d])
    iis_1d = np.array(
        np.concatenate(
            np.array(
                [
                    list(
                        itertools.repeat(first_dimension_iis[i],
                        iis_2d_lengths[i]))
                    for i in range(len(iis_2d_lengths))])), dtype=int)
    return (iis_1d, np.concatenate(iis_2d)), iis_2d_lengths


def _get_iis_from_list(first_dimension, second_dimension):
    """Given the indices of the first dimension, the second dimension
    (as a list), and the lengths of the ragged dimension, returns the
    2D indices and the new lengths in the ragged dimension."""
    iis = np.array(
        list(itertools.product(first_dimension, second_dimension))).T
    new_lengths = list(
        itertools.repeat(len(second_dimension), len(first_dimension)))
    return iis, new_lengths


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
    starts : array, [n]
        The indices of the 1d array that correspond to the first element in
        _array.
    """

    __slots__ = ('_data', '_array', 'lengths')

    def __init__(self, array, lengths=None, error_checking=True, copy=True):
        # Check that input is proper (array of arrays)
        if error_checking:
            array = np.array(list(array))
            if len(array) > 20000:
                # lenghts is None => we are not inferring lengths from
                # e.g. nested lists
                if lengths is None:
                    logger.warning(
                        "error checking is turned off for ragged arrays "
                        "with first dimension greater than 20000")
            else:
                _ensure_ragged_data(array)

        # prepare self._data
        if (len(array) > 0) and (lengths is None):
            logger.debug("Interpreting array as list/array of lists/arrays.")
            if _is_iterable(array[0]):
                if not copy:
                    warnings.warn(
                        "Can't create a view into %s, copying anyway." %
                        type(array), RuntimeWarning)
                self._data = np.concatenate(array)
            else:
                self._data = np.array(array, copy=copy)
        elif len(array) > 0:
            logger.debug("Interpreting array as concatenated array.")
            self._data = np.array(array, copy=copy)

        # Prepare with _array
        # new array greater with >0 elements
        if (lengths is None) and (len(array) > 0):
            # array of arrays
            if _is_iterable(array[0]):
                self.lengths = np.array([len(i) for i in array], dtype=int)
                self._array = np.array(
                    partition_list(self._data, self.lengths), dtype='O')
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
            try:
                self._array = np.array(
                    partition_list(self._data, lengths), dtype='O')
            except DataInvalid:
                raise DataInvalid(
                    "Sum of lengths (%s) didn't match data shape (%s)." %
                    (sum(lengths), self._data.shape))
            self.lengths = np.array(lengths)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        if np.any(self.lengths-self.lengths[0]):
            rag_second_dim = None
        else:
            rag_second_dim = self.lengths[0]
        if _is_iterable(self._data[0]):
            data_dim = self._data.shape
            if len(data_dim) == 1:
                return (len(self.lengths), rag_second_dim, None)
            else:
                return (len(self.lengths), rag_second_dim, self._data.shape[1])
        return (len(self.lengths), rag_second_dim)

    @property
    def size(self):
        return len(self._data)

    @property
    def starts(self):
        return np.append([0], np.cumsum(self.lengths)[:-1])

    # Built in functions
    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return _format_array(self._array, '__repr__')
    def __str__(self):
        return _format_array(self._array, '__str__')

    def __getitem__(self, iis):
        # ints are handled by numpy
        if isinstance(iis, numbers.Integral):
            return self._array[iis]
        # slices and lists are handled by numpy, but return a RaggedArray
        elif isinstance(iis, (slice, list, np.ndarray)):
            return RaggedArray(self._array[iis])
        # tuples get index conversion from 2d to 1d
        elif isinstance(iis, tuple):
            first_dimension, second_dimension = iis
            # if the first dimension is a slice, converts both sets of indices
            if isinstance(first_dimension, slice):
                first_dimension_iis = _slice_to_list(
                    first_dimension, length=len(self.lengths))
                # if the second dimension is a slice, determines the 2d indices
                # from the lengths in the ragged dimension
                if isinstance(second_dimension, slice):
                    iis, new_lengths  = _get_iis_from_slices(
                        first_dimension_iis, second_dimension, self.lengths)
                # if second dimension is an int, make it look like a list
                # and get iis
                elif isinstance(second_dimension, numbers.Integral):
                    iis, new_lengths = _get_iis_from_list(
                        first_dimension_iis, [second_dimension])
                else:
                    iis, new_lengths = _get_iis_from_list(
                        first_dimension_iis, second_dimension)
            elif isinstance(second_dimension, slice):
                # if the first dimension is an int, but the second is
                # a slice, numpy can handle it.
                if isinstance(first_dimension, numbers.Integral):
                    return self._array[first_dimension][second_dimension]
                # if the second dimension is a slice, determines the 2d indices
                # from the lengths in the ragged dimension
                else:
                    first_dimension_iis = first_dimension
                    iis, new_lengths  = _get_iis_from_slices(
                        first_dimension_iis, second_dimension, self.lengths)
            # If the indices are a tuple, but does not contain a slice,
            # does regular conversion.
            else:
                return self._data[
                        _convert_from_2d(
                            iis, lengths=self.lengths, starts=self.starts)]
            # Takes 2D indices generated from slicing in first or second
            #dimension and returns data formatted with new_lengths
            sliced_data = self._data[
                _convert_from_2d(
                    iis, lengths=self.lengths, starts=self.starts)]
            return RaggedArray(sliced_data, lengths=new_lengths)

        # if the indices are of self, assumes a boolean matrix. Converts
        # bool to indices and recalls __getitem__
        elif type(iis) is type(self):
            iis = where(iis)
            return self.__getitem__(iis)

    def __setitem__(self, iis, value):
        if type(value) is type(self):
            value = value._array
        # ints, slices, lists, and numpy objects are handled by numpy
        if isinstance(iis, (numbers.Integral, slice, list, np.ndarray)):
            self._array[iis] = value
            self.__init__(self._array)
        # tuples get index conversion from 2d to 1d
        elif isinstance(iis, tuple):
            first_dimension, second_dimension = iis
            # if the first dimension is a slice, converts both sets of indices
            if isinstance(first_dimension, slice):
                first_dimension_iis = _slice_to_list(
                    first_dimension, length=len(self.lengths))
                # if second dimension is a slice, determines the 2d indices
                # from the lengths in the ragged dimension
                if isinstance(second_dimension, slice):
                    iis, new_lengths = _get_iis_from_slices(
                        first_dimension_iis, second_dimension, self.lengths)
                # if the second dimension is an int, make it look like a list
                # and get iis
                elif isinstance(second_dimension, numbers.Integral):
                    iis, new_lengths = _get_iis_from_list(
                        first_dimension_iis, [second_dimension])
                else:
                    iis, new_lengths = _get_iis_from_list(
                        first_dimension_iis, second_dimension)
            elif isinstance(second_dimension, slice):
                # if the first dimension is an int, but the second is
                # a slice, numpy can handle it.
                if isinstance(first_dimension, numbers.Integral):
                    self._array[first_dimension][second_dimension] = value
                    self.__init__(self._array)
                    return
                # if the second dimension is a slice, pick the maximum length
                # of all arrays for conversion of slice to list. Indices that
                # do not exist are later removed.
                else:
                    first_dimension_iis = first_dimension
                    iis, new_lengths = _get_iis_from_slices(
                        first_dimension_iis, second_dimension, self.lengths)
            # If the indices are a tuple, but does not contain a slice,
            # does regular conversion.
            else:
                iis_1d = _convert_from_2d(
                    iis, lengths=self.lengths, starts=self.starts)
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
                    partition_list(self._data, self.lengths), dtype='O')
                return
            # Takes 2D indices generated from slicing in the first or second
            # dimension and sets data values to input values
            iis_1d = _convert_from_2d(
                iis, lengths=self.lengths, starts=self.starts)
            if _is_iterable(value):
                if _is_iterable(value[0]):
                    value_1d = np.concatenate(value)
                else:
                    value_1d = value
            else:
                value_1d = value
            self._data[iis_1d] = value_1d
            self._array = np.array(
                partition_list(self._data, self.lengths), dtype='O')
        # if the indices are of self, assumes a boolean matrix. Converts
        # bool to indices and recalls __getitem__
        elif type(iis) is type(self):
            iis = where(iis)
            self.__setitem__(iis, value)

    def __invert__(self):
        new_data = self._data.__invert__()
        return RaggedArray(new_data, lengths=self.lengths)

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
    def __or__(self, other):
        return self.map_operator('__or__', other)
    def __xor__(self, other):
        return self.map_operator('__xor__', other)
    def __and__(self, other):
        return self.map_operator('__and__', other)
    def map_operator(self, operator, other):
        if type(other) is type(self):
            other = other._data
        new_data = getattr(self._data, operator)(other)

        if new_data is NotImplemented:
            return NotImplemented
        else:
            return RaggedArray(array=new_data, lengths=self.lengths,
                               error_checking=False)

    # Non-built in functions
    def all(self):
        return np.all(self._data)

    def any(self):
        return np.any(self._data)

    def max(self):
        return self._data.max()

    def min(self):
        return self._data.min()

    @property
    def size(self):
        return self._data.size

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
                partition_list(self._data, self.lengths), dtype='O')

    def flatten(self):
        return self._data.flatten()
