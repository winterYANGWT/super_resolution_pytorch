import torch
import numpy as np
import math

class LossMeter(object):
    def __init__(self):
        super(LossMeter,self).__init__()
        self.reset()

    def reset(self):
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,count):
        self.sum+=val*count
        self.count+=count
        self.avg=self.sum/self.count

def calc_PSNR(mse):
    return 10*math.log10(1/mse)

