# MNIST数据集导出

[toc]

## 1、 MNIST数据集介绍

&emsp;&emsp;MNIST数据集是一个包含手写数字图像的大型数据库，被广泛应用于训练各种图像处理系统和机器学习模型。

**来源**
&emsp;&emsp;MNIST数据集由美国国家标准与技术研究所（National Institute of Standards and Technology，NIST）发起整理，一共统计了来自250个不同的人手写数字图片。其中50%是高中生，50%来自人口普查局的工作人员。

**数据信息**
&emsp;&emsp;MNIST数据集包含70000张图像，其中60000张用于训练，10000张用于测试。每一张图像都是28×28像素的灰度图像，代表一个手写数字。这种格式使得机器学习模型更容易识别和分类这些数字，并且能够更好地捕捉到数字的细节和纹理信息。

**应用场景**
&emsp;&emsp;MNIST数据集被广泛应用于各种图像处理和机器学习的任务中，特别是手写数字识别。它已经成为计算机视觉和深度学习领域中的一个经典数据集。许多关于神经网络的教程都会使用MNIST数据集作为例子来解释神经网络的工作原理。此外，许多研究者会使用MNIST数据集来比较和评估他们的算法和模型，并与其他研究者的结果进行比较。

**评估指标**
&emsp;&emsp;在手写数字识别任务中，常用的评估指标包括准确率、精确率、召回率和F1分数等。这些指标用于评估模型的性能，并帮助我们了解模型的优缺点。

**类别说明**
&emsp;&emsp;MNIST数据集中的每个图像都属于一个特定的类别，即手写数字。数据集中的数字类别是从0到9的整数，总共有10个不同的类别。每个类别中包含了大量的图像，以便训练模型进行分类。在训练过程中，模型需要学会将每个图像归类到相应的数字类别中，并尽可能准确地预测出数字的值。在测试过程中，模型需要对其从未见过的图像进行分类和预测，以评估其性能和准确性。

## 2、 MNIST数据集下载

### 2.1 使用Pytorch自带的MNIST数据集

&emsp;&emsp;MNIST作为一个经典数据集，Pytorch已经在datasets里自带了该数据集，可以通过pytorch数据集自动下载该数据集。

```python
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
```

&emsp;&emsp;该脚本会在data/mnist/MNIST/raw路径下下载MNIST数据集，并自动解压，生成4个二进制文件。

```
dataset_uncompressed/
├── t10k-images-idx3-ubyte                #测试集图像数据，7840016字节
├── t10k-labels-idx1-ubyte                #测试集标签数据，10008字节
├── train-images-idx3-ubyte                #训练集图像数据，47040016字节
└── train-labels-idx1-ubyte                #训练集标签数据，60008字节
```

## 3、 MNIST数据集解析

### 3.1 训练集图片文件解析规则

&emsp;&emsp;MNIST数据集中，训练集图片保存在二进制文件`train-images-idx3-ubyte`中。
|偏移量(offset)|值类型|数值|含义|
|-------|-------|-------|-------|
|0|32位整型|0x00000803|magic number|
|4|32位整型|60000|图片总数|
|8|32位整型|28|图片宽度|
|12|32位整型|28|图片高度|
|16|unsigned 8位整型|28 $*$ 28个|第0个图片数据|
|16+i $*$ 28 $*$ 28|unsigned 8位整型|28 $*$ 28个|第i个图片数据|

&emsp;&emsp;数据文件总大小47040016 = 60000 $*$ 28 $*$ 28 $+$ 4 $*$ 4.

### 3.2 训练集标签文件解析规则

&emsp;&emsp;MNIST数据集中，训练集标签保存在二进制文件`train-labels-idx1-ubyte`中。
|偏移量(offset)|值类型|数值|含义|
|-------|-------|-------|-------|
|0|32位整型|0x00000801|magic number|
|4|32位整型|60000|图片总数|
|8+|unsigned 8位整型|0-9|第i个图片的标签|

&emsp;&emsp;数据文件总大小60008 = 60000 $*$ 1 $+$ 2 $*$ 4.

### 3.3 测试集图片文件解析规则

&emsp;&emsp;MNIST数据集中，测试集图片保存在二进制文件`t10k-images-idx3-ubyte`中。
|偏移量(offset)|值类型|数值|含义|
|-------|-------|-------|-------|
|0|32位整型|0x00000803|magic number|
|4|32位整型|10000|图片总数|
|8|32位整型|28|图片宽度|
|12|32位整型|28|图片高度|
|16|unsigned 8位整型|28 $*$ 28个|第0个图片数据|
|16+i $*$ 28 $*$ 28|unsigned 8位整型|28 $*$ 28个|第i个图片数据|

&emsp;&emsp;数据文件总大小7840016 = 10000 $*$ 28 $*$ 28 $+$ 4 $*$ 4.

### 3.4 测试集标签文件解析规则

&emsp;&emsp;MNIST数据集中，测试集标签保存在二进制文件`t10k-labels-idx1-ubyte`中。
|偏移量(offset)|值类型|数值|含义|
|-------|-------|-------|-------|
|0|32位整型|0x00000801|magic number|
|4|32位整型|10000|图片总数|
|8+ i |unsigned 8位整型|0-9|第i个图片的标签|

&emsp;&emsp;数据文件总大小10008 = 10000 $\times$ 1 $+$ 2 $\times$ 4.

## 4、 MNIST数据集转图片

### 4.1 使用python手动解析MNIST数据

&emsp;&emsp;根据第3节中所述的数据解析规则，可以使用python手动解析其中的数据，并保存成png图片文件。

```python
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def load_image_data(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    magic_number, num_images, num_rows, num_cols = np.frombuffer(data[:16], np.dtype('>i4'))
    images = np.frombuffer(data[16:], np.dtype('u1')).reshape(num_images, num_rows, num_cols)
    return images

def load_label_data(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    magic_number, num_labels = np.frombuffer(data[:8], np.dtype('>i4'))
    labels = np.frombuffer(data[8:], np.dtype('u1'))
    return labels

def save_image_data(images, labels, type='train'):
    for i, (image, label) in enumerate(zip(images, labels)):
        if not os.path.exists(f'./data/mnist/{type}/{label}'):
            os.makedirs(f'./data/mnist/{type}/{label}')
        cv2.imwrite(f'./data/mnist/{type}/{label}/{i}.png', image)

def main():
    # 读取训练集数据
    train_images = load_image_data('./data/mnist/MNIST/raw/train-images-idx3-ubyte')
    train_labels = load_label_data('./data/mnist/MNIST/raw/train-labels-idx1-ubyte')
    # 读取测试集数据
    test_images = load_image_data('./data/mnist/MNIST/raw/t10k-images-idx3-ubyte')
    test_labels = load_label_data('./data/mnist/MNIST/raw/t10k-labels-idx1-ubyte')
    # 保存训练集数据
    save_image_data(train_images, train_labels, 'train')
    # 保存测试集数据
    save_image_data(test_images, test_labels, 'test')

if __name__ == '__main__':
    main()
```

&emsp;&emsp;解析出来的图片如下所示。
![alt text](images/1.png) ![alt text](images/3.png) ![alt text](images/5.png) ![alt text](images/7.png) ![alt text](images/2.png) ![alt text](images/0.png) ![alt text](images/13.png) ![alt text](images/15.png) ![alt text](images/17.png) ![alt text](images/4.png)

|数据集|标签|数目|数据集|标签|数目|
|-------|-------|-------|-------|-------|-------|
|训练集|0|5923|测试集|0|980|
|训练集|1|6742|测试集|1|1135|
|训练集|2|5958|测试集|2|1032|
|训练集|3|6131|测试集|3|1010|
|训练集|4|5842|测试集|4|982|
|训练集|5|5421|测试集|5|892|
|训练集|6|5918|测试集|6|958|
|训练集|7|6265|测试集|7|1028|
|训练集|8|5851|测试集|8|974|
|训练集|9|5949|测试集|9|1009|
|汇总||60000|||10000|

### 4.2 使用C++导出数据

```C++
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

std::vector<cv::Mat> loadMNISTImages(const std::string& path)
{
    std::ifstream file(path,std::ios::in|std::ios::binary);
    if(file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int number_of_rows = 0;
        int number_of_columns = 0;
        file.read((char*)&magic_number,sizeof(magic_number));
        file.read((char*)&number_of_images,sizeof(number_of_images));
        file.read((char*)&number_of_rows,sizeof(number_of_rows));
        file.read((char*)&number_of_columns,sizeof(number_of_columns));
        number_of_images = _byteswap_ulong(number_of_images);
        number_of_rows = _byteswap_ulong(number_of_rows);
        number_of_columns = _byteswap_ulong(number_of_columns);
        std::vector<cv::Mat> images;
        for(int i = 0; i < number_of_images; i++)
        {
            cv::Mat image(number_of_rows,number_of_columns,CV_8UC1);
            file.read((char*)image.data,number_of_rows*number_of_columns);
            images.push_back(image);
        }
        return images;
    }
    return std::vector<cv::Mat>();
}

std::vector<int> loadMNISTLabels(const std::string& path)
{
    std::ifstream file(path,std::ios::in|std::ios::binary);
    if(file.is_open())     
    {
        int magic_number = 0;
        int number_of_items = 0;
        file.read((char*)&magic_number,sizeof(magic_number));
        file.read((char*)&number_of_items,sizeof(number_of_items));
        number_of_items = _byteswap_ulong(number_of_items);
        std::vector<int> labels;
        for(int i = 0; i < number_of_items; i++)
        {
            unsigned char label = 0;
            file.read((char*)&label,sizeof(label));
            labels.push_back(label);
        }
        return labels;
    }
    return std::vector<int>();
}

void exportMNISTImages(const std::string& imagesPath, const std::string& labelsPath, const std::string& outputPath, const std::string& type)
{
    std::vector<cv::Mat> images = loadMNISTImages(imagesPath);
    std::vector<int> labels = loadMNISTLabels(labelsPath);
    std::filesystem::path output(outputPath);
    if(type == "train")
    {
        output = output / "train";
    }
    else if(type == "test")
    {
        output = output / "test";
    }
    else
        return;

    if(images.size() == labels.size())
    {
        std::filesystem::create_directory(outputPath);
        for(int i = 0; i < images.size(); i++)
        {

            std::string label = std::to_string(labels[i]);
            auto outPath = output / label;
            if(!std::filesystem::exists(outPath))
                std::filesystem::create_directories(outPath);

            std::string filename = outPath.string() + "/" + std::to_string(i) + ".png";
            cv::imwrite(filename,images[i]);
        }
    }
    return ;
}
int main(int argc, char** argv)
{
    if(argc < 6)
    {
        std::cout << "Usage: MNISTExporter <trainimagesPath> <trainlabelsPath> <testimagesPath> <testlabelsPath> <outputPath>" << std::endl;
        return 1;
    }
    auto sTrainImagesPath = std::string(argv[1]);
    auto sTrainLabelPath = std::string(argv[2]);
    auto sTestImagePath = std::string(argv[3]);
    auto sTestLabelPath = std::string(argv[4]);
    auto sOutputPath = std::string(argv[5]);
    exportMNISTImages(sTrainImagesPath,sTrainLabelPath,sOutputPath,"train");
    exportMNISTImages(sTestImagePath,sTestLabelPath,sOutputPath,"test");
    return 0;
}
```

## 5、 总结

&emsp;&emsp;MNIST数据集作为一个常用的入门深度学习的数据集，该数据集提供了60000张图片用于训练集，10000张图片用于训练集。其中，0-9共10个标签，在训练集和测试集中，各个类别的数目大致均衡。
