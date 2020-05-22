import functools
import itertools
import logging
import os
import sys
import pdb
import collections
import random
import shutil
from collections import deque
import glob
import csv

from matplotlib import pyplot as plt
import numpy as np
import torch
import joblib
from sklearn import model_selection

import mathtools as m


logger = logging.getLogger(__name__)


# --=( SCRIPT UTILITIES )=-----------------------------------------------------
def in_ipython_console():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


if in_ipython_console():
    from IPython import get_ipython
    ipython = get_ipython()


def autoreload_ipython():
    if in_ipython_console():
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")


def setupRootLogger(stream=None, filename=None, level=logging.INFO, write_mode='a'):
    """ Set up a root logger.

    Parameters
    ----------
    stream : in {sys.stdout, sys.stderr}
    filename : str
    level : in {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR}
    write_mode : str in {'a', 'w'}, optional

    Returns
    -------
    logger :
    """

    if stream is None:
        stream = sys.stdout

    logger = logging.getLogger()
    logger.handlers = []

    logging.captureWarnings(True)
    fmt_str = '[%(levelname)s] %(message)s'

    handlers = [logging.StreamHandler(stream=stream)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename, mode=write_mode))

    logging.basicConfig(
        level=level, format=fmt_str, handlers=handlers)

    return logger


def setupStreamHandler():
    root_logger = logging.getLogger()
    # Workaround deletes ipython's automatic handler
    if in_ipython_console():
        root_logger.handlers = []

    logging.captureWarnings(True)
    fmt_str = '[%(levelname)s]  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt_str)
    root_logger.handlers[0].setLevel(logging.INFO)


def makeProcessTimeStr(elapsed_time):
    """ Convert elapsed_time to a h/m/s string. """

    SECS_IN_MIN = 60
    elapsed_secs = int(round(elapsed_time % SECS_IN_MIN))
    elapsed_time //= SECS_IN_MIN

    MINS_IN_HR = 60
    elapsed_mins = int(round(elapsed_time % MINS_IN_HR))
    elapsed_time //= MINS_IN_HR

    elapsed_hrs = int(round(elapsed_time))

    time_str = '{}h {}m {}s'.format(
        elapsed_hrs,
        elapsed_mins,
        elapsed_secs
    )

    fmt_str = 'Finished. Process time: {}'
    return fmt_str.format(time_str)


def exceptionHandler(exc_type, exc_value, exc_traceback):
    """ Log non-KeyboardInterrupt exceptions, then start pdb in postmortem mode. """
    # Ignore KeyboardInterrupt so a console python program can exit with
    # Ctrl + C
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    pdb.pm()


def setupExceptionHandler():
    if in_ipython_console():
        logger.info('Using iPython\'s exception handler')
    else:
        logger.info('Not in iPython -- using custom exception handler')
        sys.excepthook = exceptionHandler


def getUniqueIds(dir_path, prefix=None, suffix=None):
    if prefix is None:
        prefix = "trial-"
    if suffix is None:
        suffix = ".pkl"

    trial_ids = set(
        int(os.path.basename(fn).split(prefix)[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"{prefix}*{suffix}"))
    )

    return np.array(sorted(tuple(trial_ids)))


def writeResults(results_file, metric_dict, sweep_param_name, model_params):
    rows = []

    if not os.path.exists(results_file):
        header = list(metric_dict.keys())
        if sweep_param_name is not None:
            header = [sweep_param_name] + header
        rows.append(header)

    row = list(metric_dict.values())
    if sweep_param_name is not None:
        param_val = model_params[sweep_param_name]
        row = [param_val] + row
    rows.append(row)

    with open(results_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


# -=( CROSS VALIDATION )==-----------------------------------------------------
def makeDataSplits(dataset_size, val_ratio=None, precomputed_fn=None, **kwargs):
    """ Splits data into cross-validation folds.

    Parameters
    ----------
    dataset_size : int
    val_ratio : float, optional
        Proportion of the training data that should be used for validation.
    precomputed_fn : string, optional
        If this argument is provided, pre-computed splits will be loaded from a file.

    Returns
    -------
    data_splits : tuple( tuple(data) )
        Each element of `data_splits` is tuple of data that represents a
        particular cross-validation fold. In other words,
        `data_splits[i] = (train_set_i, val_set_i, test_set_i)`
    """

    if precomputed_fn is not None:
        splits = joblib.load(precomputed_fn)
        return splits

    if not kwargs:
        cv_splitter = model_selection.LeaveOneOut()
    else:
        cv_splitter = model_selection.KFold(**kwargs)

    data_indices = np.arange(dataset_size)
    splits = cv_splitter.split(data_indices)

    if val_ratio is not None:
        def makeVal(i_train, i_test):
            num_val = roundToInt(val_ratio * len(i_train))
            i_val = i_train[:num_val]
            i_train = i_train[num_val:]
            return i_train, i_val, i_test
        splits = tuple(makeVal(i_train, i_test) for i_train, i_test in splits)

    return splits


def makeDataSplitFromRatio(*data, test_ratio=0.2, val_ratio=None):
    """ Split data into train, test, and validation sets using provided data ratios.

    Parameters
    ----------
    data : iterable
        Each argument should be an iterable of data.
    test_ratio : float, optional
        Proportion of the data that should be used as the test set.
    val_ratio : float, optional
        Proportion of the data that should be used as the validation set.
    test_set : int, optional
        Indices of data samples in the test set. Useful for leave-one-out cross
        validation.
    val_size : int, optional
        Indices of data samples in the validation set.

    Returns
    -------
    train_set : tuple(data)
        Data samples to use for training.
    val_set : tuple(data)
        Data samples to use for validation.
    test_set : tuple(data)
        Data samples to use for testing.
    """

    num_samples = len(data[0])

    indicators = makeExptSplits(num_samples, test_ratio, val_ratio)

    def selectData(data, indices):
        if isinstance(data, np.ndarray):
            return data[indices]
        else:
            return filterSeqs(data, indices)

    data_splits = tuple(
        tuple(selectData(d, ind) for ind in indicators)
        for d in data
    )

    return tuple(zip(*data_splits))


def makeDataSplitFromSets(*data, test_set, val_set=None):
    """ Split data into train, test, and validation sets using a provided test set.

    Parameters
    ----------
    data : iterable
        Each argument should be an iterable of data.
    test_ratio : float, optional
        Proportion of the data that should be used as the test set.
    val_ratio : float, optional
        Proportion of the data that should be used as the validation set.
    test_set : int, optional
        Indices of data samples in the test set. Useful for leave-one-out cross
        validation.
    val_size : int, optional
        Indices of data samples in the validation set.

    Returns
    -------
    train_set : tuple(data)
        Data samples to use for training.
    val_set : tuple(data)
        Data samples to use for validation.
    test_set : tuple(data)
        Data samples to use for testing.
    """

    num_samples = len(data[0])

    idxs = np.arange(num_samples)
    is_test = arrayMatchesAny(idxs, test_set)
    is_train = np.logical_not(is_test)
    is_val = np.zeros(num_samples, dtype=bool)
    indicators = (is_train, is_val, is_test)

    data_splits = tuple(
        tuple(filterSeqs(d, ind) for ind in indicators)
        for d in data
    )
    return tuple(zip(*data_splits))


def makeExptSplits(num_samples, test_ratio, val_ratio=None):
    sample_idxs = np.arange(num_samples)
    np.random.shuffle(sample_idxs)

    num_test_samples = int(num_samples * test_ratio)
    is_test = isInSplit(sample_idxs, num_test_samples)
    offset = num_test_samples

    if val_ratio is not None:
        num_val_samples = int(num_samples * val_ratio)
        is_val = isInSplit(sample_idxs, num_val_samples, offset)
        offset += num_val_samples

    is_train = isInSplit(sample_idxs, offset=offset)

    if val_ratio is None:
        return is_train, is_test

    return is_train, is_val, is_test


def isInSplit(sample_idxs, num_split_samples=None, offset=0):
    num_samples = sample_idxs.max() + 1

    if num_split_samples is None:
        end = None
    else:
        end = offset + num_split_samples
    split_idxs = sample_idxs[offset:end]

    is_in_split = np.zeros(num_samples, dtype=bool)
    is_in_split[split_idxs] = True

    return is_in_split


def validateCvFold(train_idxs, test_idxs):
    """ Raise an error if the train and test sets overlap.

    Parameters
    ----------
    train_idxs : iterable(int)
    test_idxs : iterable(int)

    Raises
    ------
    ValueError
        When any element is common to both the train and test sets.
    """

    intersect = set(train_idxs) & set(test_idxs)

    if intersect:
        err_str = f"Indices {intersect} occur in training and test sets!"
        raise ValueError(err_str)


# -=( INTEGERIZERS )==---------------------------------------------------------
class Integerizer(object):
    """ Collection associating distinct object types with consecutive integers.

    NOTE: based on https://github.com/seq2class/assignment3/blob/master/seq2class_homework2.py
    NOTE 2: objects must be hashable
    """

    def __init__(self, iterable=[]):
        """ Initialize the collection.

        Initializes to the empty set, or to the set of *unique* objects in its
        argument (in order of first occurrence).
        """
        # Set up a pair of data structures to convert objects to ints and back again.
        self._objects = []   # list of all unique objects that have been added so far
        self._indices = {}   # maps each object to its integer position in the list
        # Add any objects that were given.
        self.update(iterable)

    def __len__(self):
        """ Number of objects in the collection. """
        return len(self._objects)

    def __iter__(self):
        """ Iterate over all the objects in the collection. """
        return iter(self._objects)

    def __contains__(self, obj):
        """ Does the collection contain this object?  (Implements `in`.) """
        return self.index(obj) is not None

    def __getitem__(self, index):
        """ Return the object with a given index.

        (Implements subscripting, e.g., `my_integerizer[3]`.)
        """
        return self._objects[index]

    def _getIndex(self, obj):
        return self._indices[obj]

    def index(self, obj, add=False):
        """ The integer associated with a given object.

        Returns `None` if the object is not in the collection (OOV).
        Use `add=True` to add the object if it is not present.
        """
        try:
            return self._getIndex(obj)
        except KeyError:
            if add:
                return self.append(obj)
            else:
                return None

    def add(self, obj):
        """ Add the object if it is not already in the collection.

        Similar to `set.add`.
        """
        return self.index(obj, add=True)

    def append(self, obj):
        """ Append the object, regardless of whether it is already in the collection.

        Similar to `list.append`.
        """
        self._objects.append(obj)
        self._indices[obj] = len(self) - 1
        return self._indices[obj]

    def update(self, iterable):
        """ Add all the objects if they are not already in the collection.

        Similar to `set.update` (or `list.extend`).
        """
        for obj in iterable:
            self.add(obj)


class UnhashableIntegerizer(Integerizer):
    """ Integerize unhashable objects. """

    def _getIndex(self, obj):
        """ This method emulates a dict, but actually just does list lookups. """
        # for i, candidate in enumerate(self._objects):
        #     if obj == candidate:
        #         return i
        # else:
        #     raise KeyError
        try:
            return self._objects.index(obj)
        except ValueError:
            # Make this list lookup raise errors like a dict
            raise KeyError

    def append(self, obj):
        """ Append the object, regardless of whether it is already in the collection.

        Similar to `list.append`.
        """
        self._objects.append(obj)
        return len(self) - 1


# --=( GRAPH )=----------------------------------------------------------------
def dfs(start_index, edges, visited=None):
    """ Generate indices of nodes encountered during a depth-first search starting from `index`.

    Parameters
    ----------
    start_index : int
        Index of the vertex to begin the search at.
    edges : numpy array of bool, shape (num_vertices, num_vertices)
        The graph's adjacency matrix.
    visited : numpy array of bool, shape (num_vertices,), optional
        The i-th element is `True` if the i-th vertex has already been visited.
        This is useful for e.g. finding all the connected components of a graph.

    Yields
    ------
    index : int
        Index of the next unvisited vertex encountered.
    """

    if visited is None:
        visited = m.np.zeros(edges.shape[0], bool)

    stack = [start_index]

    while stack:
        index = stack.pop()
        if visited[index]:
            continue

        visited[index] = True
        neighbors = edges[index,:].nonzero()[0].tolist()
        stack += neighbors
        yield index


def bfs(start_index, edges, visited=None):
    """ Generate indices of nodes encountered during a breadth-first search starting from index.

    Parameters
    ----------
    start_index : int
        Index of the vertex to begin the search at.
    edges : numpy array of bool, shape (num_vertices, num_vertices)
        The graph's adjacency matrix.
    visited : numpy array of bool, shape (num_vertices,), optional
        The i-th element is `True` if the i-th vertex has already been visited.
        This is useful for e.g. finding all the connected components of a graph.

    Yields
    ------
    index : int
        Index of the next unvisited vertex encountered.
    """

    if visited is None:
        visited = m.np.zeros(edges.shape[0], bool)

    queue = deque
    queue.appendleft(start_index)

    while queue:
        index = queue.pop()
        if visited[index]:
            continue

        visited[index] = True
        neighbors = edges[index,:].nonzero()[0].tolist()
        # NOTE: extendleft reverses the order of neighbors, but that's OK
        #   in this situation
        queue.extendleft(neighbors)
        yield index


# --=( STRING )=---------------------------------------------------------------
def stripExtension(file_path):
    """ Return a copy of `file_path` with the file extension removed. """

    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def strToBool(string):
    """ Convert `"True"` and `"False"` to their boolean counterparts. """

    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        err_str = f"{string} is not a boolean value!"
        raise ValueError(err_str)


def camelCase(string):
    capitalized = ''.join(string.title().split())
    return capitalized[0].lower() + capitalized[1:]


# --=( SEQUENCE )=-------------------------------------------------------------
def genSegments(seq):
    cur_state = seq[0]
    segment_len = 0
    for state in seq[1:]:
        segment_len += 1
        if state != cur_state:
            yield cur_state, segment_len
            cur_state = state
            segment_len = 0
    else:
        segment_len += 1
        yield cur_state, segment_len


def nearestIndex(sequence, val, seq_sorted=False):
    """ Find the index of the element in sequence closest to val. """

    # Handle corner case: length-1 sequence
    if len(sequence) == 1:
        return 0

    if seq_sorted:
        return nearestIndexSorted(sequence, val)

    return nearestIndexUnsorted(sequence, val)


def nearestIndexSorted(sequence, val):
    """ Return the index in `sequence` which is closest to `val` in absolute value.

    Parameters
    ----------
    sequence : numpy.array
        Sorted numpy array.
    val : float or int
        The reference value.

    Returns
    -------
    idx : int
        Index of the item in `sequence` which is closest to `val` in absolute value.
    """

    idx = m.np.searchsorted(sequence, val, side="left")
    prev_idx = idx - 1

    if idx == 0:
        return idx

    if idx == len(sequence):
        return prev_idx

    prev_closer = abs(val - sequence[prev_idx]) < abs(val - sequence[idx])
    if prev_closer:
        return prev_idx

    return idx


def nearestIndexUnsorted(sequence, val):
    def distance(sequence, index):
        return abs(sequence[index] - val)

    dist = functools.partial(distance, sequence)
    seq_indices = range(len(sequence))
    return min(seq_indices, key=dist)


def nearestIndices(sequence, values):
    """ Return the indices in `sequence` nearest to each items in `values`.

    NOTE: `sequence` and `values` must be sorted in increasing order.

    Parameters
    ----------
    sequence : numpy.array
    values : iterable

    Returns
    -------
    indices : numpy array of int, shape (num_values,)
        1-D array whose elements correspond to the elements in `values`.
        The i-th element of `indices` is the index of the element in `sequence`
        which is closest to the i-th element of `values` in absolute value.
    """

    indices = []
    last_index = 0
    for val in values:
        cur_seq = sequence[last_index:]
        last_index += nearestIndex(cur_seq, val, seq_sorted=True)
        indices.append(last_index)

    return np.array(indices)


def signalEdges(sequence, edge_type=None):
    """ Detect edges in a binary signal.

    Parameters
    ----------
    sequence : numpy array of bool, shape (sum_samples,)
    edge_type : {'rising', 'falling'}
        If 'rising', returns edges transitioning from 0 to 1.
        If 'falling', returns edges transitioning from 1 to 0.
        If `None`, returns both types.

    Returns
    -------
    edge_idxs : tuple(int)
        Indices of edges in the input.
    """

    difference = m.np.diff(sequence.astype(int))

    if edge_type is None:
        is_edge = difference != 0
    elif edge_type == 'rising':
        is_edge = difference > 0
    elif edge_type == 'falling':
        is_edge = difference < 0

    edge_idxs = is_edge.nonzero()[0]
    return edge_idxs


def arrayMatchesAny(source_array, target_set):
    """ Elementwise implementation of `source_array in target_set`.

    Parameters
    ----------
    source_array : numpy array, shape (NUM_SAMPLES,)
    target_set : iterable

    Returns
    -------
    in_set : numpy array of bool, shape (NUM_SAMPLES,)
        The i-th element of `in_set` is `True` if the i-th element of
        `source_array` is found in `target_set`.
    """

    matches_single_target = (source_array == target for target in target_set)
    matches_any = functools.reduce(m.np.logical_or, matches_single_target)
    return matches_any


def isEmpty(obj):
    """ Check if an object is empty. The object could be a numpy array,
    an iterable object, etc. """
    if isinstance(obj, m.np.ndarray):
        return not obj.any()
    return not obj


def findEmptyIndices(iterable):
    empty_indices = tuple(
        i for i, elem in enumerate(iterable)
        if isEmpty(elem)
    )
    return empty_indices


def filterSeqs(seqs_to_filter, ref_seqs, filter_func=isEmpty):
    """ filter first arg based on truth value of second arg. """
    pairs = zip(seqs_to_filter, ref_seqs)
    return tuple(x for x, y in pairs if not filter_func(y))


def resampleSeq(sample_seq, sample_times, new_times):
    """ Resample a signal by choosing the nearest sample in time. """

    if isEmpty(new_times):
        warn_str = 'Encountered empty resampling array'
        logger.warning(warn_str)

    resampled_indices = nearestIndices(sample_times, new_times)

    if isinstance(sample_seq, np.ndarray):
        return sample_seq[resampled_indices]
    else:
        return tuple(sample_seq[i] for i in resampled_indices)


def drawRandomSample(samples, sample_times, start_time, end_time):
    """ Draw a sample uniformly at random from the samples between start_time and end_time. """
    start_index = nearestIndex(sample_times, start_time, seq_sorted=True)
    end_index = nearestIndex(sample_times, end_time, seq_sorted=True)

    sample_index = m.np.random.randint(start_index, end_index + 1)
    return samples[sample_index]


def computeWindowLength(window_duration, sample_rate):
    """ Convert window duration (units of time) to window length (units of samples).

    NOTE: This assumes uniform sampling.
    """
    return round(window_duration * sample_rate)


def computeStrideLength(window_length, overlap_ratio):
    """
    Convert window length (units of samples) and window overlap ratio to stride
    length (number of samples between windows). This assumes uniform sampling.
    """
    return round(window_length * (1 - overlap_ratio))


def splitSeq(seq, predicates):
    """
    Split a sequence into one seq for which predicates is True, and another
    for which predicates is False.

    Parameters
    ----------
    seq : iterable
      An arbitrary sequence.
    predicates : iterable( bool )
      Same length as seq. Each entry is the truth value of some predicate
      evaluated on seq.

    Returns
    -------
    pred_true : iterable
      Elements in seq whose corresponding entries in predicates are True
    pred_false : iterable
      Elements in seq whose corresponding entries in predicates are False
    """

    seq_len = len(seq)
    pred_len = len(predicates)
    if seq_len != pred_len:
        err_str = '{} sequence elements but {} predicates!'
        raise ValueError(err_str.format(seq_len, pred_len))

    pred_true = tuple(elem for elem, pred in zip(seq, predicates) if pred)
    pred_false = tuple(elem for elem, pred in zip(seq, predicates) if not pred)

    return pred_true, pred_false


def concatDataDict(data_dict_1, data_dict_2):
    if not data_dict_2:
        return data_dict_1

    if not data_dict_1:
        return data_dict_2

    dict_1_keys = data_dict_1.keys()
    dict_2_keys = data_dict_2.keys()

    if dict_1_keys != dict_2_keys:
        err_str = 'Argument dicts do not contain the same data!'
        raise ValueError(err_str)

    concat_dict = {}
    for key in dict_1_keys:
        val = data_dict_1[key] + data_dict_2[key]
        concat_dict[key] = val

    return concat_dict


def sampleDataDict(sample_indices, data_dict):
    for key, value in data_dict.items():
        data_dict[key] = drawSamples(value, sample_indices)

    return data_dict


def randomSample(num_items, num_samples):
    indices = tuple(range(num_items))
    sampled_indices = random.sample(indices, num_samples)
    return sampled_indices


def drawSamples(sequence, indices):
    return tuple(sequence[i] for i in indices)


def select(indices, sequence):
    return drawSamples(sequence, indices)


def slidingWindow(signal, window_len, stride_len=1, axis=-1, padding=None):
    """ Create a tensor representing a sliding window over the input signal.

    Parameters
    ----------
    signal : numpy array, shape (..., num_samples)
    window_len : int
    stride_len : int, optional
    padding : {'reflect', 'zero', None}, optional

    Returns
    -------
    windows : numpy array, shape (..., num_samples, num_windows)
    """

    if not window_len % 2:
        raise NotImplementedError('Only odd-length windows are currently supported')

    if stride_len != 1:
        raise NotImplementedError('Only unit stride is currently supported')

    if axis != -1:
        signal = np.moveaxis(signal, axis, -1)
        windows = slidingWindow(
            signal, window_len,
            stride_len=stride_len, axis=-1, padding=padding
        )
        return np.moveaxis(windows, -2, axis)

    if padding == 'reflect':
        half_window = window_len // 2
        pre_padding = signal[..., half_window:0:-1]
        post_padding = signal[..., -2:-2 - half_window:-1]
        signal = np.concatenate((pre_padding, signal, post_padding), axis=-1)
    elif padding == 'zero':
        padding = np.zeros_like(signal[..., :half_window])
        signal = np.concatenate((padding, signal, padding), axis=-1)
    elif padding is not None:
        raise ValueError('argument "padding={padding}" is not supported')

    shape = signal.shape[:-1] + (signal.shape[-1] - window_len + 1, window_len)
    strides = signal.strides + (signal.strides[-1],)
    windows = np.lib.stride_tricks.as_strided(
        signal, shape=shape, strides=strides,
        writeable=False
    )
    return windows


def extractWindows(signal, window_size=10, return_window_indices=False):
    """ Reshape a signal into a series of non-overlapping windows.

    Parameters
    ----------
    signal : numpy array, shape (num_samples,)
    window_size : int, optional
    return_window_indices : bool, optional

    Returns
    -------
    windows : numpy array, shape (num_windows, window_size)
    window_indices : numpy array of int, shape (num_windows, window_size)
    """

    tail_len = signal.shape[0] % window_size
    pad_arr = m.np.full(window_size - tail_len, m.np.nan)
    signal_padded = m.np.concatenate((signal, pad_arr))
    windows = signal_padded.reshape((-1, window_size))

    if not return_window_indices:
        return windows

    indices = m.np.arange(signal_padded.shape[0])
    window_indices = indices.reshape((-1, window_size))

    return windows, window_indices


def computeSampleTimes(sample_rate, start_time, end_time):
    sample_period = 1 / sample_rate

    # np.arange can sometimes return values beyond end_time
    sample_times = np.arange(start_time, end_time, sample_period)
    sample_times = sample_times[sample_times <= end_time]

    return sample_times


# -=( SEGMENTS, LABELS AND STUFF)==--------------------------------------------
def computeSegments(seq):
    segments, segment_lens = zip(*genSegments(seq))
    return tuple(segments), tuple(segment_lens)


def makeSegmentLabels(label_seq):
    seg_labels, seg_lens = computeSegments(label_seq)

    seg_label_seq = np.zeros_like(label_seq)
    start_index = 0
    for i, seg_len in enumerate(seg_lens):
        end_index = start_index + seg_len
        seg_label_seq[start_index:end_index] = i
        start_index = end_index

    return seg_label_seq


def reduce_all_equal(segment):
    if not np.all(segment == segment[0]):
        raise AssertionError()
    return segment[0]


def reduce_over_segments(signal, seg_label_seq, reduction=None):
    if reduction is None:
        reduction = reduce_all_equal

    segment_labels = np.unique(seg_label_seq)
    num_segments = segment_labels.shape[0]
    num_samples = signal.shape[0]
    remaining_dims = signal.shape[1:]

    samples = np.zeros((num_samples,) + remaining_dims)
    segments = np.zeros((num_segments,) + remaining_dims)

    for i in segment_labels:
        in_segment = seg_label_seq == i
        segment = signal[in_segment]
        reduced = reduction(segment)

        samples[in_segment] = reduced
        segments[i] = reduced

    return segments, samples


# --=( FUNCTIONAL PROGRAMMING )=-----------------------------------------------
def mapValues(function, dictionary):
    """ Map `function` to the values of `dictionary`. """
    return {key: function(value) for key, value in dictionary.items()}


def compose(*functions):
    """ Compose functions left-to-right.

    NOTE: This implementation is based on a blog post by Mathieu Larose.
        https://mathieularose.com/function-composition-in-python/

    Parameters
    ----------
    *functions : function
        functions = func_1, func_2, ... func_n
        These should be single-argument functions. If they are not functions of
        a single argument, you can do partial function application using
        `functools.partial` before calling `compose`.

    Returns
    -------
    composition : function
        Composition of the arguments. In other words,
        composition(x) = func_1( func_2( ... func_n(x) ... ) )
    """

    def compose2(f, g):
        return lambda x: f(g(x))

    composition = functools.reduce(compose2, functions)

    return composition


def pipeline(*functions):
    """ Compose functions right-to-left.

    This function mimics a data pipeline:
        input --> pipeline(function 1, function 2, ..., function n) --> output
    can be thought of as an implementation of
        input --> function 1 --> function 2 --> ... --> function n --> output

    Parameters
    ----------
    *functions : function
        functions = func_1, func_2, ... func_n
        These should be single-argument functions. If they are not functions of
        a single argument, you can do partial function application using
        `functools.partial` before calling `compose`.

    Returns
    -------
    composition : function
        Composition of the arguments. In other words,
        composition(x) = func_n( ... func_2( func_1(x) ) ... )
    """

    return compose(*reversed(functions))


def batchProcess(
        function, *input_sequences,
        obj=tuple, unzip=False,
        static_args=None, static_kwargs=None):

    if all(x is None for x in input_sequences):
        return None

    # with ProcessPoolExecutor() as executor:
    #     output_sequence = executor.map(function, *input_sequences)
    # return list(output_sequence)

    if static_args is not None:
        function = functools.partial(function, *static_args)

    if static_kwargs is not None:
        function = functools.partial(function, **static_kwargs)

    ret = obj(map(function, *input_sequences))

    if unzip:
        return obj(zip(*ret))

    return ret


def zipValues(*dicts):
    """ Iterate over dictionaries with the same keys, ensuring the values are
    always ordered the same way. """

    first_keys = tuple(dicts[0].keys())

    # Make sure all dicts have the same keys
    for d in dicts[1:]:
        d_keys = tuple(d.keys())
        if d_keys != first_keys:
            err_str = (
                f"can't zip dicts because keys differ: "
                "{d_keys} != {first_keys}"
            )
            raise ValueError(err_str)

    # Iterating like this ensures the correspondence between keys is preserved
    zipped_vals = tuple(tuple(d[key] for d in dicts) for key in first_keys)
    return tuple(zip(*zipped_vals))


def countItems(items):
    """ Count each unique element occuring in `items`.

    Parameters
    ----------
    items : iterable
        It is assumed that all elements of `items` have the same type.

    Returns
    -------
    counts : collections.Counter
    """

    counts = collections.Counter()

    if isinstance(items[0], collections.abc.Iterable):
        for item in items:
            counts.update(countItems(item))
    else:
        counts.update(items)

    return counts


# --=( NUMERICAL COMPUTATIONS )=-----------------------------------------------
def plotArray(data, fn=None, label=None):
    """ Plot an array with accompanying colorbar. """

    num_rows, num_cols = data.shape

    fig, ax = plt.subplots()
    cax = ax.imshow(data, aspect='auto')

    if label is not None:
        ax.set_title(label)

    if num_rows > num_cols:
        fig.colorbar(cax, pad=0.05)
    else:
        fig.colorbar(cax, orientation='horizontal', pad=0.05)

    fig.tight_layout()

    if fn is not None:
        plt.savefig(fn)
        plt.close()


def nanargmax(signal, axis=1):
    """ Select the greatest non-Nan entry from each row.

    If a row does not contain at least one non-NaN entry, it does not have a
    corresponding output.

    Parameters
    ----------
    signal : numpy array, shape (num_rows, num_cols)
        The reference signal for argmax. This is usually an array constructed
        by windowing a univariate signal. Each row is a window of the signal.
    axis : int, optional
        Axis along which to compute the argmax. Not implemented yet.

    Returns
    -------
    non_nan_row_idxs : numpy array of int, shape (num_non_nan_rows,)
        An array of the row indices in `signal` that have at least one non-NaN
        entry, arranged in increasing order.
    non_nan_argmax : numpy array of int, shape (num_non_nan_rows,)
        An array of the greatest non-NaN entry in `signal` for each of the rows
        in `non_nan_row_idxs`.
    """

    if axis != 1:
        err_str = 'Only axis 1 is supported right now'
        raise NotImplementedError(err_str)

    # Determine which rows contain all NaN
    row_is_all_nan = m.np.isnan(signal).all(axis=1)
    non_nan_row_idxs = m.np.nonzero(~row_is_all_nan)[0]

    # Find the (non-NaN) argmax of each row with at least one non-NaN value
    non_nan_rows = signal[~row_is_all_nan, :]
    non_nan_argmax = m.np.nanargmax(non_nan_rows, axis=1)

    return non_nan_row_idxs, non_nan_argmax


def argmaxNd(array):
    """ Return the array indices that jointly maximize the input array.

    Parameters
    ----------
    array : numpy array, shape (n_dim_1, n_dim_2, ..., n_dim_k)

    Returns
    -------
    argmax : tuple(int), length-k
    """

    argmax = m.np.unravel_index(array.argmax(), array.shape)
    return argmax


def safeDivide(numerator, denominator):
    """ Safe divide function that handles division by zero.

    Parameters
    ----------
    numerator : float
    denominator : float

    Returns
    -------
    quotient : float
        ``numerator / denominator``. If both `numerator` and `denominator` are
        zero, returns 0. If only `denominator` is zero, returns ``m.np.inf``.
    """

    if not numerator and not denominator:
        return 0

    if not denominator:
        return m.np.inf

    return numerator / denominator


def makeHistogram(
        num_classes, samples, normalize=False, ignore_empty=True, backoff_counts=0):
    """ Make a histogram representing the empirical distribution of the input.

    Parameters
    ----------
    samples : numpy array of int, shape (num_samples,)
        Array of outcome indices.
    normalize : bool, optional
        If True, the histogram is normalized to it sums to one (ie this
        method returns a probability distribution). If False, this method
        returns the class counts. Default is False.
    ignore_empty : bool, optional
        Can be useful when ``normalize == False`` and ``backoff_counts == 0``.
        If True, this function doesn't try to normalize the histogram of an
        empty input. Trying to normalize that histogram would lead to a
        divide-by-zero error and an output of all ``NaN``.
    backoff_counts : int or float, optional
        This amount is added to all bins in the histogram before normalization
        (if `normalize` is True). Non-integer backoff counts are allowed.

    Returns
    -------
    class_histogram : numpy array of float, shape (num_clusters,)
        Histogram representing the input data. If `normalize` is True, this is
        a probability distribution (possibly smoothed via backoff). If not,
        this is the number of times each cluster was encountered in the input,
        plus any additional backoff counts.
    """

    hist = np.zeros(num_classes)

    # Sometimes the input can be empty. We would rather return a vector of
    # zeros than trying to normalize and get NaN.
    if ignore_empty and not np.any(samples):
        return hist

    # Count occurrences of each class in the histogram
    for index in range(num_classes):
        class_count = np.sum(samples == index)
        hist[index] = class_count

    # Add in the backoff counts
    hist += backoff_counts

    if normalize:
        hist /= hist.sum()

    return hist


def boolarray2int(array):
    return sum(1 << i for i, b in enumerate(array) if b)


def int2boolarray(integer, num_objects):
    bool_array = m.np.zeros(num_objects, dtype=bool)

    bool_list = []
    while integer:
        bool_list.append(integer % 2)
        integer = integer >> 1

    start = -1
    end = -1 - len(bool_list)
    stride = -1
    bool_array[start:end:stride] = bool_list

    return bool_array


def roundToInt(X):
    """ Round a numpy array to the nearest integer.

    Parameters
    ----------
    X : numpy array

    Returns
    -------
    rounded : numpy array of int
    """

    if isinstance(X, np.ndarray):
        return m.np.rint(X).astype(int)
    if isinstance(X, torch.Tensor):
        return m.np.rint(X).int()
    return int(round(X))


def splitColumns(X):
    if len(X.shape) != 2:
        err_str = ''
        raise ValueError(err_str)

    num_cols = X.shape[1]
    return m.np.hsplit(X, num_cols)


def sampleRows(X, num_samples):
    sample_indices = m.np.random.randint(0, X.shape[0], size=num_samples)
    return X[sample_indices, :]


def isUnivariate(signal):
    """ Returns True if `signal` contains univariate samples.

    Parameters
    ----------
    signal : np.array, shape (NUM_SAMPLES, NUM_DIMS)

    Returns
    -------
    is_univariate : bool
    """

    if len(signal.shape) == 1:
        return True
    if signal.shape[1] > 1:
        return False
    return True


def lower_tri(array):
    """ Return the lower-triangular part of an array (below the main diagonal).

    Parameters
    ----------
    array : array, shape (..., N, N) or (N, N)

    Returns
    -------
    lower_tri : array, shape (..., N * (N - 1) / 2)
    """

    if array.shape[-1] != array.shape[-2]:
        raise ValueError(f"Array should be square in last two dims, but has shape {array.shape}")

    rows, cols = np.tril_indices(array.shape[-1], k=-1)
    return array[..., rows, cols]


# --=( I/O )=------------------------------------------------------------------
def loadVariable(attr_name, base_name=None, corpus_name=None):
    working_path = os.path.expanduser(
        os.path.join('~', 'repo', 'blocks', 'data', 'working')
    )
    corpus_path = os.path.join(working_path, base_name, corpus_name)
    var_fn = f'{attr_name}.pkl'
    var_path = os.path.join(corpus_path, var_fn)

    attr_val = joblib.load(var_path)

    return attr_val


def saveVariable(var, attr_name, base_name=None, corpus_name=None, overwrite=False):
    working_path = os.path.expanduser(
        os.path.join('~', 'repo', 'blocks', 'data', 'working')
    )
    corpus_path = os.path.join(working_path, base_name, corpus_name)
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)

    var_fn = f'{attr_name}.pkl'
    var_path = os.path.join(corpus_path, var_fn)
    if os.path.exists(var_path):
        if not overwrite:
            err_str = f'File {var_fn} already exists! To overwrite, call me with overwrite=True'
            raise FileExistsError(err_str)

    joblib.dump(var, var_path)


def loadVideoData(corpus_name):
    prefixes = ('rgb', 'depth')
    suffixes = ('frame_fns', 'timestamps')
    pairs = itertools.product(prefixes, suffixes)
    attr_names = tuple('_'.join(p) for p in pairs)

    frame_fns = {}
    for attr_name in attr_names:
        frame_fns[attr_name] = loadVariable(attr_name, corpus_name)

    return frame_fns


def copyFile(file_path, dest_dir):
    """ Copy a file to a new directory, preserving its filename.

    Parameters
    ----------
    file_path : str
        Path to the file that will be copied.
    dest_dir : str
        Path to the directory where the copy will be placed.

    Returns
    -------
    dest_path : str
        Path to the copied file.
    """

    file_dir, file_fn = os.path.split(file_path)
    dest_path = os.path.join(dest_dir, file_fn)
    shutil.copy(file_path, dest_path)

    return dest_path


# --=( VISUALIZATION )=--------------------------------------------------------
def plot_array(inputs, labels, label_names, fn=None, tick_names=None):
    subplot_width = 12
    subplot_height = 3

    num_axes = 1 + len(labels)

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize, sharex=True)

    inputs = inputs.reshape(-1, inputs.shape[-1])
    axes[-1].imshow(inputs, interpolation='none', aspect='auto')
    axes[-1].set_ylabel('Input')

    for i, (label, label_name) in enumerate(zip(labels, label_names)):
        if label.ndim == 1:
            axes[i].plot(label, label=label_name)
        elif label.ndim == 2:
            axes[i].imshow(label, interpolation='none', aspect='auto')
        if tick_names is not None:
            axes[i].set_yticks(range(len(tick_names)))
            axes[i].set_yticklabels(tick_names)
        axes[i].set_ylabel(label_name)

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()
