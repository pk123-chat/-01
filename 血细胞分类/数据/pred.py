from dataset import *
import torch
import matplotlib.pyplot as plt

# 加载模型
model = torch.load('Bloodcell.pkl', map_location=torch.device('cpu'))  # 强制将模型加载到CPU

# 选取一些图片进行预测
img, label = next(iter(dl_test))

# 使用 CPU 而不是 CUDA
img = img.to('cpu')  # 将图像数据转移到 CPU
model.eval()  # 设置模型为评估模式
pred = model(img)  # 模型预测
pred_re = torch.argmax(pred, dim=1)  # 获取预测结果的类别索引

# 将预测结果转换为列表
pred_re = pred_re.cpu().numpy()  # 将预测结果移回 CPU
pred_re = pred_re.tolist()  # 转换为列表

# 打印预测结果的类别
for i in pred_re[0:8]:
    print(id_to_class[i])
print(id_to_class[pred_re[0:8][1]])

# 显示图像与预测结果
plt.figure(figsize=(16, 8))
img = img.cpu()  # 把图片重新放到CPU上
for i, (img, label) in enumerate(zip(img[:8], label[:8])):
    img = (img.permute(1, 2, 0).numpy() + 1) / 2  # 归一化处理
    plt.subplot(2, 4, i + 1)
    pred_title = id_to_class[pred_re[0:8][i]]  # 获取预测标题
    plt.title('R:{}, P:{}'.format(id_to_class.get(label.item()), pred_title))  # 显示真实值和预测值
    plt.imshow(img)  # 显示图像

plt.show()  # 展示所有图像
