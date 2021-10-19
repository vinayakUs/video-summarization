from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable
from pathlib import Path
import cv2
import numpy as np

"""
pre-trained ResNet
"""


class ResNet(nn.Module):
    """
    Args:
        fea_type: string, resnet101 or resnet 152
    """

    def __init__(self, fea_type='resnet152'):
        super(ResNet, self).__init__()
        self.fea_type = fea_type
        # rescale and normalize transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if fea_type == 'resnet101':
            resnet = models.resnet101(pretrained=True)  # dim of pool5 is 2048
        elif fea_type == 'resnet152':
            resnet = models.resnet152(pretrained=True)
        else:
            raise Exception('No such ResNet!')

        resnet.float()
        resnet.cuda()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]

    # rescale and normalize image, then pass it through ResNet
    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)  # reshape the single image s.t. it has a batch dim
        x = Variable(x).cuda()
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)

        return res_pool5


class VideoProcessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.resnet = ResNet()

    def get_features(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # can change the model to googleNet
                feat = self.resnetfeatures(frame)
                features.append(feat)
            n_frames += 1

        cap.release()

        features = np.array(features)
        return n_frames, features, fps

    def run(self, video_path: Path):
        n_frames, features, fps = self.get_features(video_path)
        return n_frames, features, fps
