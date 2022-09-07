import os
import re
import shutil


def copy_file(src, dst):
    """
    Copy source (src) file to destination (dst) file.

    Note that renaming is possible.

    Args:
        src: source file to be copied to the destination file
        dst: destination file (potentially renamed) from source file.

    Returns:

    """
    shutil.copy(src, dst)


def create_folder_if_not_exists(folder, logger=None):
    """
    Create folder if it does not exist yet.

    It is also possible to create subfolders. For example, if path D:/foo exists and we want to create D:/foo/bar/baz,
    but D:/foo/bar does not exist, then it is still possible to directly create D:/foo/bar/baz by
    create_folder_if_not_exists('D:/foo/bar/baz').

    Args:
        folder:
        logger:

    Returns:

    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if logger is not None:
        logger.my_print('Creating folder: {}'.format(folder))


def sort_human(l):
    """
    Sort the input list. However normally with l.sort(), e.g., l = ['1', '2', '10', '4'] would be sorted as
    l = ['1', '10', '2', '4']. The sort_human() function makes sure that l will be sorted properly,
    i.e.: l = ['1', '2', '4', '10'].
    
    Source: https://stackoverflow.com/questions/3426108/how-to-sort-a-list-of-strings-numerically

    Args:
        l: to-be-sorted list

    Returns:
        l: properly sorted list
    """
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

