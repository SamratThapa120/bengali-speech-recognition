import numpy as np



class FrequencyMasking:
    def __init__(self, mask_ratio=(0,0.05), num_masks=2,prob=0.2):
        self.min_ratio = mask_ratio[0]
        self.max_ratio = mask_ratio[1]
        self.num_masks = num_masks
        self.prob = prob
    def __call__(self, spec):
        if np.random.rand()<self.prob:
            for _ in range(self.num_masks):
                f = int(np.random.uniform(low=self.min_ratio, high=self.max_ratio)*spec.shape[0])
                f0 = np.random.uniform(low=0, high=spec.shape[0]-f)
                spec[int(f0):int(f0+f), :] = 0
        return spec

class TimeMasking:
    def __init__(self, mask_ratio=(0,0.01), num_masks=2,prob=0.2):
        self.min_ratio = mask_ratio[0]
        self.max_ratio = mask_ratio[1]
        self.num_masks = num_masks
        self.prob = prob
    def __call__(self, spec):
        if np.random.rand()<self.prob:
            for _ in range(self.num_masks):
                f = int(np.random.uniform(low=self.min_ratio, high=self.max_ratio)*spec.shape[1])
                f0 = np.random.uniform(low=0, high=spec.shape[1]-f)
                spec[:,int(f0):int(f0+f)] = 0
        return spec

