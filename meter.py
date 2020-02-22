class AverageMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,count):
        self.sum+=val*count
        self.count+=count
        self.avg=self.sum/self.count

