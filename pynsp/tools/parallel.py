import psutil
from pynsp.base import np
from joblib import Parallel, delayed


def optimal_split(num, cpus, scale=10):
    result = [num%(i+1) for i in range(cpus*scale)]
    return [i+1 for i, x in enumerate(result) if x == 0][-1]


def avail_cpu():
    cpu_percent = np.sum(psutil.cpu_percent(percpu=True))
    cpu_count = psutil.cpu_count()
    return int(cpu_count * (1 - ((cpu_percent / cpu_count) * 2)/100))