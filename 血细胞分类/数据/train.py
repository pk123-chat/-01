import torch as t
import torch.nn as nn
from tqdm import tqdm  #进度条
import net
from dataset import *
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

# 数据集
train_dataset = torchvision.datasets.ImageFolder(
  'E:/Jupytercode/血细胞分类/数据/blood-cells/dataset2-master/dataset2-master/images/TRAIN',
  transform=train_transformer
)
test_dataset = torchvision.datasets.ImageFolder(
  'E:/Jupytercode/血细胞分类/数据/blood-cells/dataset2-master/dataset2-master/images/TEST',
  transform=test_transformer
)

# 类别ID映射
id_to_class = {}
for k, v in train_dataset.class_to_idx.items():
    id_to_class[v] = k

# 数据加载
Batch_size = 64
dl_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=Batch_size,
    shuffle=True
)
dl_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=Batch_size,
    shuffle=True
)

# 初始化模型
model = net.Net()
model = model.to(device)  # 将模型转移到设备上

optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器

loss_fn = nn.CrossEntropyLoss()

# 训练和验证过程
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()  # 训练模式下  识别normalize层
    for x, y in tqdm(trainloader):
        x, y = x.to(device), y.to(device)  # 使用动态选择的设备
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()  # 验证模式下   不识别normalize层
    with torch.no_grad():
        for x, y in tqdm(testloader):
            x, y = x.to(device), y.to(device)  # 使用动态选择的设备
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    if epoch_acc > 0.95:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, './{}{}.pth'.format(epoch_acc, epoch_test_acc))

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

if __name__ == '__main__':
    epochs = 10
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                     model,
                                                                     dl_train,
                                                                     dl_test)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

    plt.plot(range(1, epochs + 1), train_loss, label='train_loss')  # 绘制训练损失曲线
    plt.plot(range(1, epochs + 1), test_loss, label='test_loss')  # 绘制验证损失曲线
    plt.legend()  # 添加图例
    plt.xlabel('Epochs')  # 设置横坐标轴标签
    plt.ylabel('Loss')  # 设置纵坐标轴标签
    plt.savefig('loss.png')
    plt.show()

    plt.plot(range(1, epochs + 1), train_acc, label='train_acc')
    plt.plot(range(1, epochs + 1), test_acc, label='test_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc.png')
    plt.show()

    torch.save(model, 'Bloodcell.pkl')  # 保存模型训练权重
