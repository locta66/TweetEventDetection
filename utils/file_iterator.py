import os
import shutil
import utils.pattern_utils as pu


def add_sep_if_needed(path): return path if path.endswith(os.path.sep) else path + os.path.sep


def pwd(): return os.getcwd()


def base_name(abspath): return os.path.basename(abspath)


def get_parent_path(path): return os.path.dirname(path)


def is_dir(path): return os.path.isdir(path)


def is_file(file): return os.path.isfile(file)


def exists(path_or_file): return os.path.exists(path_or_file)


def makedirs(path): os.makedirs(path)


def rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def remove_file(file):
    if type(file) is str:
        if os.path.exists(file):
            os.remove(file)
    else:
        raise TypeError("File descriptor not an expected type")


def remove_files(files):
    if type(files) is list:
        for f in files:
            remove_file(f)
    else:
        raise TypeError("File descriptor array not an expected type")


def make_dirs(dir_name):
    if not type(dir_name) is str:
        raise ValueError('Not valid directory description token')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def concat_files(file_list, output_file):
    input_files = ' '.join(file_list)
    p = os.popen('cat {} > {}'.format(input_files, output_file))
    p.close()


TYPE_DIR = 'dir'
TYPE_FILE = 'file'
TYPE_ALL = 'all'


def listchildren(directory, children_type=TYPE_DIR, pattern=None):
    if children_type not in [TYPE_DIR, TYPE_FILE, TYPE_ALL]:
        print('listchildren() : Incorrect children type')
        return None
    directory = add_sep_if_needed(directory)
    children = sorted(os.listdir(directory))
    if pattern is not None:
        children = [c for c in children if pu.search_pattern(pattern, c) is not None]
    if children_type == TYPE_ALL:
        return children
    res = list()
    for child in children:
        child_path = directory + child
        if not os.path.exists(child_path):
            print('listchildren() : Invalid path')
            continue
        _is_dir = os.path.isdir(child_path)
        if children_type == TYPE_DIR and _is_dir:
            res.append(child)
        elif children_type == TYPE_FILE and not _is_dir:
            res.append(child)
    return res


def iterate_file_tree(root_path, func, *args, **kwargs):
    """
    Iterate file tree under the root path,, and invoke func in those directories with no sub-directories.
    :param root_path: Current location of tree iteration.
    :param func: A callback function.
    :param args: Just to pass over outer params for func.
    :param kwargs: Just to pass over outer params for func.
    """
    subdirs = listchildren(root_path, children_type='dir')
    if not subdirs:
        func(root_path, *args, **kwargs)
    else:
        for subdir in subdirs:
            iterate_file_tree(root_path + subdir + os.path.sep, func, *args, **kwargs)
