from pathlib import Path
from pandas import read_csv, DataFrame
from re import compile
from numpy import asarray, savetxt, ndarray
from sys import argv
from utilities.pre_processing import final_clean

file_regex = compile(r'\w+\.\w+$')
type_map = {
    'O': '%s',
    'i': '%d',
    'f': '%.6f'
}


def make_path(filename):
    """
    Makes path from given string

    :param str filename: Filename of target file
    :return Path: Path of target file
    """
    return Path(filename)


def check_existence(paths):
    """ Checks whether the file(s) exist """
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if isinstance(path, str):
            path = make_path(path)

        if not isinstance(path, Path):
            raise TypeError('Provided path', path, 'is not of type Path.')
        if not path.exists():   # Check if file exists
            raise FileExistsError(path, 'does not exist.')


def load_data(path, columns=None, index_col=0, encoding='utf-8'):
    """
    Opens file as a Panda DataFrame

    :param Path path: Path to file
    :param list columns: List of column names to import the data with [uses top row by default]
    :param int index_col: Index of column to use as index values [default first column]
    :param str encoding: Encoding of the file [guesses by default]
    :return DataFrame: DataFrame containing file content
    """
    data_frame = read_csv(path, usecols=columns, index_col=index_col, encoding=encoding)

    return data_frame


def output_abusive_intent(indexes, predictions, contexts, filename=None):
    """
    Prints abusive intent results to console and saves to disk

    :param Iterator indexes: Array of targeted indexes to output (ex. output from argsort)
    :param ndarray predictions: Array of predictions (3 x N array)
    :param ndarray contexts: Array of corresponding documents
    :param Path filename: Path for predictions to be saved to [doesn't save by default]
    """
    indexes = asarray(list(indexes)) if not isinstance(indexes, ndarray) else indexes
    abuse, intent, abusive_intent = predictions

    if filename is not None:
        header = ','.join(['abuse', 'intent', 'abusive_intent'])
        savetxt(filename, indexes, delimiter=',', fmt='%d', header=header)

    print('%10s %8s %8s %8s  %s' % ('index', 'hybrid', 'intent', 'abuse', 'context'))
    for index in indexes:
        print(
            '%10d %8.4f %8.4f %8.4f  %s' % (index, abusive_intent[index], intent[index], abuse[index], contexts[index])
        )


def save_vector(data_vector, path):
    """
    Saves a numpy vector to a csv without indexes or column header(s)

    :param ndarray data_vector: Array of values
    :param Path path: Path to save array to
    """
    data_type = data_vector.dtype.kind
    if data_type not in type_map:
        raise TypeError('Unsupported type,', data_type)

    savetxt(path, data_vector, delimiter=',', fmt=type_map[data_type])


def load_vector(file_path):
    """
    Loads data from a csv with a single column and no header

    :param Path file_path: Path to target file
    :return ndarray: Data from file
    """
    file_data = read_csv(file_path, header=None)\
        .values.reshape(-1)

    return file_data


def check_execution_targets():
    """ Checks if the python arguments specify valid file paths for consumption """
    targets = [Path(target) for target in argv[1:]] if len(argv) > 1 else [None]

    for target in targets:
        if target is None or not target.exists():
            print('Specified data does not exist, using environment target.')
            return False
    return True


def save_contexts(contexts, indexes, target_path):
    """
    Save document contexts and indexes relating them to their parent documents
    
    :param list contexts: List of contexts extracted from documents
    :param ndarray indexes: Array of indexes for each context and its parent document
    :param Path target_path: Destination path for contexts
    """
    dataset = DataFrame(indexes, columns=['document_index', 'context_index'])

    dataset['contexts'] = list(map(final_clean, contexts))
    dataset.to_csv(target_path)
