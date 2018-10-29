from .sampen import SampEn
import numpy as np

data = np.random.randint(-50, 50, 900)

print(SampEn(data, 3, 0.2))