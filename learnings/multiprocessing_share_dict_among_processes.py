"""
https://stackoverflow.com/a/55816091
"""
from itertools import repeat
import multiprocessing as mp
import os
import pprint


def worker(index, shared_dict):
    pid = os.getpid()
    shared_dict[pid] = 'hola!'
    print(shared_dict['name'])


def predict():
    with mp.Manager() as manager:
        # shared dictionaries among all processes
        shared_dict = {'name': 'PACO'}
        shared_dict = manager.dict(shared_dict)

        with manager.Pool(processes=2) as pool:
            pool.starmap(worker, [(1, shared_dict), (2, shared_dict)])

        # `d` is a DictProxy object that can be converted to dict
        pprint.pprint(dict(shared_dict))


predict()