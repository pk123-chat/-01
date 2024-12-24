import torch.nn as nn
import torch
#建立神经网络
class Net(nn.Module):  # 模仿VGG
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.shape)

        # 将原来的张量 x (四维)重新塑造为一个二维张量。第一个维度的大小由 PyTorch 自动计算，而第二个维度的大小被设置为 256 * 14 * 14
        x = x.view(-1, 256 * 14 * 14)
        x = self.fc(x)

        return x
if __name__ == '__main__':
    x = torch.rand([8, 3, 256, 256])
    model = Net()
    y = model(x)

