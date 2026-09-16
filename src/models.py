"""
CIFAR-adapted architectures for the calibration-under-compression project.

Standard ImageNet ResNets start with a 7x7 stride-2 conv + maxpool,
which is designed to aggressively downsample 224x224 inputs. Applied to
32x32 CIFAR images, that would shrink them to near-nothing before the
network does any real work. The well-established CIFAR adaptation
(used in the original ResNet paper's CIFAR experiments, and in nearly
every CIFAR-ResNet implementation since) replaces this stem with a
single 3x3 stride-1 conv and no maxpool.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetCifar(nn.Module):
    """
    ResNet with the CIFAR stem (3x3 conv, no maxpool).
    layers=[2,2,2,2] -> ResNet-18 (8 BasicBlocks, matches the paper's count
    once you include the stem: this is the standard "ResNet-18 for CIFAR").
    layers=[1,1,1,1] -> a shallower "ResNet-10" used as the KD student.
    """

    def __init__(self, layers, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet18_cifar(num_classes=10):
    return ResNetCifar([2, 2, 2, 2], num_classes=num_classes)


def resnet10_cifar(num_classes=10):
    """Shallower student for knowledge distillation."""
    return ResNetCifar([1, 1, 1, 1], num_classes=num_classes)


# ---------------------------------------------------------------------------
# MobileNetV2, CIFAR-adapted
#
# Standard MobileNetV2 also assumes 224x224 ImageNet input and downsamples
# by 32x overall. For 32x32 CIFAR input, we reduce the first conv's stride
# from 2 to 1, which is the standard CIFAR adaptation used across the
# efficient-architecture literature.
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Cifar(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # t, c, n, s  (expand_ratio, out_channels, num_blocks, stride)
        # first-stage stride changed 1->1 (unchanged) and overall the
        # network's first conv stride changed 2->1 below; this keeps
        # spatial resolution reasonable for 32x32 input.
        cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],   # stride 2->1: CIFAR adaptation
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult))

        features = [
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),  # stride 2->1: CIFAR adaptation
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in cfgs:
            out_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, out_channel, stride, expand_ratio=t))
                input_channel = out_channel

        features += [
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True),
        ]
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenetv2_cifar(num_classes=10, width_mult=1.0):
    return MobileNetV2Cifar(num_classes=num_classes, width_mult=width_mult)


if __name__ == "__main__":
    dummy = torch.randn(4, 3, 32, 32)  # batch of 4, matches CIFAR input shape

    print("--- resnet18_cifar ---")
    m = resnet18_cifar()
    out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  output shape: {tuple(out.shape)}  (expect (4, 10))")
    print(f"  parameter count: {n_params:,}  (expect ~11,173,962)")

    print("\n--- resnet10_cifar (KD student) ---")
    m10 = resnet10_cifar()
    out10 = m10(dummy)
    n_params10 = sum(p.numel() for p in m10.parameters())
    print(f"  output shape: {tuple(out10.shape)}  (expect (4, 10))")
    print(f"  parameter count: {n_params10:,}  (should be well under resnet18's count)")

    print("\n--- mobilenetv2_cifar ---")
    mv2 = mobilenetv2_cifar()
    out_mv2 = mv2(dummy)
    n_params_mv2 = sum(p.numel() for p in mv2.parameters())
    print(f"  output shape: {tuple(out_mv2.shape)}  (expect (4, 10))")
    print(f"  parameter count: {n_params_mv2:,}  (expect ~2.2-2.4M)")

    print("\nAll models instantiate and forward-pass correctly.")
