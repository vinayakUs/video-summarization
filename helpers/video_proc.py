from resnet import ResNet

class VideoPreprocessor(object):
    def __init__(self,sample_rate:int) -> None:
        self.sample_rate=sample_rate
        self.resnet =ResNet()