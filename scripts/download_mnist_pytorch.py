from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 下载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('./data/mnist', download=True, train=True, transform=transform)
testset = datasets.MNIST('./data/mnist', download=True, train=False, transform=transform)

# 可视化数据集图像
n = 10  # 展示10张图像
plt.figure(figsize=(10, 5))
for i in range(n):
    images, labels = trainset[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(images[0].view(28, 28), cmap='gray')
    plt.title(f'Label: {labels}')
plt.show()
