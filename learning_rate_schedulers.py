import matplotlib.pyplot as plt
import numpy as np
from config import Config

config = Config()
class LearningRateDecay:
    def plot(self, epochs, title="learning Rate Schedule"):
        lrs = [self(i) for i in epochs]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylable("Learning Rate")

class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha = config.LEARNING_RATE, factor = config.FACTOR, dropEvery = config.DROP_EVERY):
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch,optim):
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        for param_group in optim.param_groups:
            param_group['lr'] = alpha
