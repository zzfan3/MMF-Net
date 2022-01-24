import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    if save_dir:
        file_name = name + ".txt"
        fh = logging.FileHandler(os.path.join(save_dir, file_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger