# WideResNet-28-2 Model (credits: ChatGPT)

import torch

class BasicBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.3):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_planes)
        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)


        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = torch.nn.functional.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(torch.nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10):
        super().__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0  # Depth must be 6n+4
        num_blocks = (depth - 4) // 6

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16 * widen_factor, num_blocks, stride=1)
        self.layer2 = self._make_layer(32 * widen_factor, num_blocks, stride=2)
        self.layer3 = self._make_layer(64 * widen_factor, num_blocks, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(64 * widen_factor)
        self.fc = torch.nn.Linear(64 * widen_factor, num_classes)

    def _make_layer(self, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, stride))
            self.in_planes = out_planes
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.relu(self.bn1(out))
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        latent = torch.flatten(out, 1)
        out = self.fc(latent)
        return out, latent