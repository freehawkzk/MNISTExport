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