
from torchvision import transforms
import torchvision
import torch
import matplotlib.pyplot as plt
from PIL import Image
#一、数据转换
train_transformer=transforms.Compose(
[
   transforms.RandomHorizontalFlip(0.2),
   transforms.RandomRotation(68),
   transforms.RandomGrayscale(0.2),
   transforms.Resize((256,256)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5])
]
)
test_transformer=transforms.Compose(
[
   transforms.Resize((256,256)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5])
]
)
#二、读入数据
train_dataset=torchvision.datasets.ImageFolder(
  'E:/Jupytercode/血细胞分类/数据/blood-cells/dataset2-master/dataset2-master/images/TRAIN',
    transform=train_transformer
)

test_dataset=torchvision.datasets.ImageFolder(
  'E:/Jupytercode/血细胞分类/数据/blood-cells/dataset2-master/dataset2-master/images/TEST',
   transform=test_transformer
)

#进行编码
#原      {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}
#转换后  {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
id_to_class={}
for k,v in train_dataset.class_to_idx.items():
    #print(k,v)
    id_to_class[v]=k
#id_to_class #查看转换后的格式

#三、批次读入数据，可以作为神经网络的输入  一次性拿多少张图片进行训练
Batch_size=64#一次性训练64张
dl_train=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True
)
dl_test=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Batch_size,
        shuffle=True
)
#取一个批次的数据
# img,label=next(iter(dl_train))
# plt.figure(figsize=(12,8))
# for i,(img,label) in enumerate(zip(img[:8],label[:8])):
#     img=(img.permute(1,2,0).numpy()+1)/2
#     plt.subplot(2,4,i+1)
#     plt.title(id_to_class.get(label.item())) #0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'
#     plt.imshow(img)
# plt.show() #查看图片

print("数据处理已完成")

