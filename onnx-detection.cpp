#include <cpu_provider_factory.h>
// #include <cuda_provider_factory.h>

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class Image
{
  public:
    Image(int w, int h, int c)
    {
      img_buffer.reserve(w * h * c);
      apl_buffer.reserve(w * h * c);
      img_w = w;
      img_h = h;
      img_c = c;
    }

    std::vector<unsigned char> img_buffer;

    unsigned char at(int a)
    {
      return img_buffer[a];
    }

    void set(int a, unsigned char val)
    {
      img_buffer[a] = val;
    }

  private:
    std::vector<unsigned char> apl_buffer;
    int img_h;
    int img_w;
    int img_c;
};

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << "[";
  for (int i = 0; i < v.size(); ++i)
  {
      os << v[i];
      if (i != v.size() - 1)
      {
          os << ", ";
      }
  }
  os << "]";

  return os;
}

std::vector<std::string> ReadLabels(const std::string& labelsFile)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelsFile);
    while (std::getline(fp, line))
        labels.push_back(line);

    return labels;
}

int main(int argc, char* argv[])
{
  bool useCUDA{true};
  const char* useCUDAFlag = "--use_cuda";
  const char* useCPUFlag = "--use_cpu";

  if (argc == 1)
  {
      useCUDA = false;
  }
  else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
  {
      useCUDA = true;
  }
  else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
  {
      useCUDA = false;
  }
  else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0))
  {
      useCUDA = false;
  }
  else
  {
      throw std::runtime_error{"Too many arguments."};
  }

  if (useCUDA)
  {
      std::cout << "Inference Execution Provider: CUDA" << std::endl;
  }
  else
  {
      std::cout << "Inference Execution Provider: CPU" << std::endl;
  }

  const double confidenceThreshold = 0.5;
  const double maskThreshold = 0.6;

  const std::string modelFilepath = "data/models/model.onnx";
  const std::string imageFilepath = "data/images/test.jpg";
  const std::string labelFilepath = "data/labels/COCO_classes.txt";
  const std::string instanceName = "image-classification-inference";

  const std::vector<std::string> labels = ReadLabels(labelFilepath);

  int imageWidth, imageHeight, imageChannels;
  stbi_uc * img_data = stbi_load(imageFilepath.c_str(), &imageWidth,
      &imageHeight, &imageChannels, STBI_default);

  if (imageWidth != 1200 || imageHeight != 1200 || imageChannels != 3)
  {
    printf("Image size does't match with 416 x 416 x 3");
    return 1;
  }

  struct Pixel { unsigned char RGBA[3]; };
  const Pixel* imgPixels((const Pixel*)img_data);

  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
      instanceName.c_str());
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);

  // Use CUDA backend.
  if (useCUDA)
  {
    /* OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA( */
    /*     sessionOptions, 0); */
  }

  // Sets graph optimization level:
  // ORT_DISABLE_ALL - Disable all optimizations.
  // ORT_ENABLE_BASIC - Enable basic optimizations (redundant node removals).
  // ORT_ENABLE_EXTENDED - To enable extended optimizations (redundant node
  //     removals + node fusions).
  // ORT_ENABLE_ALL - Enable all possible optimizations.
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

  Ort::AllocatorWithDefaultOptions allocator;

  size_t numInputNodes = session.GetInputCount();
  std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;

  size_t numOutputNodes = session.GetOutputCount();
  std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

  const char* inputName = session.GetInputName(0, allocator);
  std::cout << "Input Name: " << inputName << std::endl;

  Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
  std::vector<int64_t> inputDims =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

  // Change the first dimension from -1 to 1, necessary for Tiny YOLO v2.
  if (inputDims[0] < 0)
    inputDims[0] = 1;
  std::cout << "Input Dimensions: " << inputDims << std::endl;

  std::cout << "Output Name: " << std::endl;
  std::vector<const char*> outputNames;
  for (size_t i = 0; i < session.GetOutputCount(); ++i)
  {
    std::cout << session.GetOutputName(i, allocator) << std::endl;
    outputNames.push_back(session.GetOutputName(i, allocator));
  }

  Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

  std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
  // Change the first dimension from -1 to 1, necessary for Tiny YOLO v2.
  if (outputDims[0] < 0)
    outputDims[0] = 1;
  std::cout << "Output Dimensions: " << outputDims << std::endl;

  // The model expects the image to be of size 416 x 416 x 3.
  const size_t inputTensorSize = imageWidth * imageHeight * imageChannels;
  std::vector<float> inputTensorValues(inputTensorSize);

  std::cout << inputTensorSize << std::endl;

  // Transpose image.
  Image *img = new Image(imageWidth, imageHeight, 3);
  size_t shift = 0;
  std::vector<float> mean{0.485, 0.456, 0.406};
  std::vector<float> std{0.229, 0.224, 0.225};
  for (size_t c = 0; c < 3; ++c)
  {
    for (size_t y = 0; y < imageHeight; ++y)
    {
      for (size_t x = 0; x < imageWidth; ++x, ++shift)
      {
        const int val(imgPixels[y * imageWidth + x].RGBA[c]);
        img->set((y * imageWidth + x) * 3 + c, val);
        inputTensorValues[shift] = (val / 255 - mean[c]) / std[c];
      }
    }
  }

  std::vector<const char*> inputNames{inputName};

  std::vector<Ort::Value> inputTensors;

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
      inputDims.size()));

  std::vector<Ort::Value> out = session.Run(Ort::RunOptions{nullptr},
      inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputNames.size());

  float* bboxes = out[0].GetTensorMutableData<float>();
  int* labelsIndex = out[1].GetTensorMutableData<int>();
  float* scores = out[2].GetTensorMutableData<float>();
  for (int i = 0; i < out[0].GetTensorTypeAndShapeInfo().GetShape()[1]; ++i)
  {
    if (scores[i] > 0.1)
    {
      std::cout << "score: " << scores[i] << std::endl;
      std::cout << "label: " << labelsIndex[i] << std::endl;
      std::cout << "["
                << bboxes[i * 4] << ", "
                << bboxes[i * 4 + 1] << ", "
                << bboxes[i * 4 + 2] << ", "
                << bboxes[i * 4 + 3]
                << "]" << std::endl;
    }
  }

  return 0;
}
