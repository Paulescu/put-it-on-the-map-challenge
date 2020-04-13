import multiprocessing
import time

def worker(x):
    time.sleep(1)
    return x

pool = multiprocessing.Pool()
print(pool.map(worker, range(10)))