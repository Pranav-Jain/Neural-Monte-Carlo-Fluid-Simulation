import os
import shutil

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    # else:
        # inp = input("File " + path + " Exists. Press y to overwrite: ")
        # if inp == "y":
        #     shutil.rmtree(path)
        #     os.makedirs(path)
        # else:
        #     print("Not overwriting")


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)
