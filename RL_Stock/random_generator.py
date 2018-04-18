
import numpy as np
from tgym.core import DataGenerator


class RandomGenerator(DataGenerator):
    @staticmethod
    def _generator(ba_spread=0):
        while True:
            val = np.random.randn()
            yield val, val + ba_spread


time_series_length = 10
mygen = RandomGenerator()
prices_time_series = [mygen.next() for _ in range(time_series_length)]
print prices_time_series