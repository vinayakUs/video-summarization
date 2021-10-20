from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable
from pathlib import Path
import cv2
import numpy as np
from src.kts.cpd_auto import cpd_auto

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

    def resnet_features(self, frame):
        frame = cv2.resize(frame, (224, 224))
        res_pool5 = self.resnet(frame)
        frame_feat = res_pool5.cpu().data.numpy().flatten()

        return frame_feat

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
                feat = self.resnet_features(frame)
                features.append(feat)
            n_frames += 1

        cap.release()

        features = np.array(features)
        return n_frames, features, fps

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: Path):
        n_frames, features, fps = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        assert isinstance(picks, object)
        return n_frames, features, fps, cps, nfps, picks

