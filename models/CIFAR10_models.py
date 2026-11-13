import torch
import torch.nn as nn
from torchvision import models


class GlobalModelForCifar10(nn.Module):
    def __init__(self, args):
        super(GlobalModelForCifar10, self).__init__()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)
        self.args = args

    def forward(self, input_list):
        tensor_t = torch.cat((input_list[0], input_list[1]), dim=1)

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.classifier(x)
        return x


class LocalModelForCifar10(nn.Module):
    def __init__(self, args):
        super(LocalModelForCifar10, self).__init__()
        self.args = args
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        if self.args.client_num == 2:
            self.backbone.fc = nn.Linear(num_ftrs, 128)

    def forward(self, x):
        x = self.backbone(x)
        return x


class SingleModelForCifar10(nn.Module):
    def __init__(self, args):
        super(SingleModelForCifar10, self).__init__()
        self.args = args
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.backbone(x)
        return x
