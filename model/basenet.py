from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


class GradReverse(torch.autograd.Function):
    """
        重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None  # neg()将每个元素都乘-1，实现梯度反转


def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.classifier[1].in_features = 2304
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, coeff=1, reverse=False):
        if reverse:
            x = grad_reverse(x, coeff)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out
