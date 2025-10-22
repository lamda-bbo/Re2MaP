import os

import logging

def prepare_directory(file_path):
    dirname = os.path.dirname(file_path)
    if dirname != "" and not os.path.exists(dirname):
        logging.info('prepare directory {}'.format(os.path.dirname(file_path)))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

def skip(func):
    def wrapped(*args, **kwargs):
        pass
    return wrapped