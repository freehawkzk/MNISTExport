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
