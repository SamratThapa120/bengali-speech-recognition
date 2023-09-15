import numpy as np



class FrequencyMasking:
    def __init__(self, F=30, num_masks=1,prob=0.2):
        self.F = F
        self.num_masks = num_masks
        self.prob = prob
    def __call__(self, spec):
        if np.random.rand()<self.prob:
            for _ in range(self.num_masks):
                f = np.random.uniform(low=0, high=self.F)
                f0 = np.random.uniform(low=0, high=spec.shape[0]-f)
                spec[int(f0):int(f0+f), :] = 0
        return spec

class TimeMasking:
    def __init__(self, T=40, num_masks=1,prob=0.2):
        self.T = T
        self.num_masks = num_masks
        self.prob =prob
    def __call__(self, spec):
        if np.random.rand()<self.prob:
            for _ in range(self.num_masks):
                t = np.random.uniform(low=0, high=self.T)
                t0 = np.random.uniform(low=0, high=spec.shape[1]-t)
                spec[:, int(t0):int(t0+t)] = 0
        return spec

