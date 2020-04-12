import os
import shutil
import pickle as pkl


def check_if_file_exists(path):
    return os.path.isfile(path)


def check_if_exists(path):
    return os.path.exists(path)


def create_nonexistent_folder(save_dir, clear_prev=False, verbose = False):
    try:
        os.makedirs(save_dir)
        return 1
    except FileExistsError:
        if clear_prev:
            import glob
            files = glob.glob(save_dir + '*')
            for f in files:
                os.remove(f)
            return 1
        else:
            if verbose:
                print('already created folder, skipping...')
            return 0


def file_modification_time(path):
    return os.path.getmtime(path)


def load_pickle_object(path):
    if check_if_file_exists(path) and path[-3:] == 'pkl':
        with open(path, 'rb') as f:
            return pkl.load(f)
    elif not check_if_file_exists(path):
        raise FileNotFoundError('Failed to find file : ' + path)


def dump_pickle_object(obj, path):

    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def delete_directory_recursively(path):

    # Delete all contents of a directory using shutil.rmtree() and  handle exceptions
    try:
        shutil.rmtree(path)
    except:
        print('Error while deleting directory')


def write_str_to_text_file(save_path, save_name, output_str):

    if '.txt' not in save_name:
        save_name += '.txt'
    with open(save_path.rstrip('/') + '/' + save_name, 'w') as f:
        f.write(output_str)
