import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, img_dim=784, num_classes=10):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(img_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)


class LogReg(nn.Module):

    def __init__(self, img_dim=784, num_classes=10):
        super(LogReg, self).__init__()
        self.net = torch.nn.Sequential(nn.Linear(img_dim, num_classes), )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)


class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [B, 128]
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
