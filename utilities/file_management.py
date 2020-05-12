from os import listdir, chdir
from pathlib import Path
from csv import field_size_limit
from sys import maxsize

root_dir_file = '.gitignore'


def move_to_root(max_levels=4, target=root_dir_file):
    """
    Changes PWD to root project directory.
    NOTE: By default assumes there is only one .gitignore file in project files.
    :param max_levels: Maximum number of directory levels checked, (default 3)
    :param target: Target file, must only be one in project files, (default .gitignore)
    """
    for i in range(max_levels):
        files = listdir('.')
        if target in files:
            return

        chdir('../')
    raise FileNotFoundError('Could not find .gitignore file, is the current dir more than', max_levels, 'deep?')


def in_parent_dir():
    """ Checks whether the PWD is the project dir """
    return Path('config.json').is_file()


def make_dir(directory, max_levels=2):
    """ Moves up directory levels until it exists, then moves back down creating child directories """
    if type(directory) is str: directory = Path(directory)
    if directory.suffix != '': directory = directory.parent
    if directory.exists(): return

    levels = [directory] + list(directory.parents)
    for ind, level in enumerate(levels):
        if ind > max_levels: break
        if level.exists():
            for l_ind in range(ind-1, -1, -1):
                levels[l_ind].mkdir()
            return

    raise LookupError('Cannot make dir within', max_levels, 'levels of the specified directory')


def expand_csv_row_size():
    """ Enables the csv reader to accept longer document rows """
    max_size = maxsize
    while True:
        try:
            field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)
