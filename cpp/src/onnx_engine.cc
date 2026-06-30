#include "cuebee/embedding_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace cuebee {
namespace {

class OnnxEngine final : public EmbeddingEngine {
 public:
  OnnxEngine(const std::string& model_path, std::int32_t intra_op_threads)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "cuebee-speaker-worker"),
        session_(nullptr) {
    if (!std::filesystem::is_regular_file(model_path)) {
      throw std::invalid_argument("ONNX model does not exist: " + model_path);
    }
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (intra_op_threads > 0) {
      options.SetIntraOpNumThreads(intra_op_threads);
    }
#ifdef _WIN32
    const std::wstring native_path = std::filesystem::path(model_path).wstring();
    session_ = Ort::Session(environment_, native_path.c_str(), options);
#else
    session_ = Ort::Session(environment_, model_path.c_str(), options);
#endif
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
      throw std::invalid_argument("ONNX model must expose exactly one input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_.GetInputNameAllocated(0, allocator);
    const auto output_name = session_.GetOutputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    output_name_ = output_name.get();

    const auto input_type_info = session_.GetInputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const ONNXTensorElementDataType input_type = input_info.GetElementType();
    if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::invalid_argument(
          "ONNX input tensor must contain float32 values; received element type " +
          std::to_string(static_cast<int>(input_type)));
    }
    const auto input_shape = input_info.GetShape();
    if (input_shape.size() != 3 ||
        (input_shape[2] > 0 && input_shape[2] != kExpectedMelBins)) {
      throw std::invalid_argument("ONNX input must have shape [batch, time, 80]");
    }
  }

  std::vector<EmbeddingResult> Infer(
      const std::vector<FeatureMatrix>& features) override {
    if (features.empty()) {
      return {};
    }
    std::uint32_t maximum_frames = 0;
    for (const FeatureMatrix& matrix : features) {
      if (matrix.frames == 0 ||
          matrix.values.size() !=
              static_cast<std::size_t>(matrix.frames) * kExpectedMelBins) {
        throw std::invalid_argument("invalid feature matrix shape");
      }
      maximum_frames = std::max(maximum_frames, matrix.frames);
    }

    const std::size_t item_stride =
        static_cast<std::size_t>(maximum_frames) * kExpectedMelBins;
    std::vector<float> batch(features.size() * item_stride, 0.0F);
    for (std::size_t index = 0; index < features.size(); ++index) {
      std::copy(features[index].values.begin(), features[index].values.end(),
                batch.begin() + static_cast<std::ptrdiff_t>(index * item_stride));
    }

    const std::array<std::int64_t, 3> input_shape = {
        static_cast<std::int64_t>(features.size()),
        static_cast<std::int64_t>(maximum_frames),
        static_cast<std::int64_t>(kExpectedMelBins),
    };
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory, batch.data(), batch.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
      throw std::runtime_error("ONNX model did not return one tensor");
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementCount() != features.size() * kEmbeddingDimension) {
      throw std::runtime_error("ONNX output must contain batch * 192 float32 values");
    }
    const float* output = outputs[0].GetTensorData<float>();

    std::vector<EmbeddingResult> results;
    results.reserve(features.size());
    for (std::size_t item = 0; item < features.size(); ++item) {
      EmbeddingResult result;
      result.embedding.assign(output + item * kEmbeddingDimension,
                              output + (item + 1) * kEmbeddingDimension);
      double norm_squared = 0.0;
      for (float value : result.embedding) {
        if (!std::isfinite(value)) {
          throw std::runtime_error("ONNX output contains a non-finite value");
        }
        norm_squared += static_cast<double>(value) * value;
      }
      const double norm = std::sqrt(norm_squared);
      if (!(norm > 1e-12)) {
        throw std::runtime_error("ONNX output contains a zero-length embedding");
      }
      for (float& value : result.embedding) {
        value = static_cast<float>(value / norm);
      }
      result.quality = static_cast<float>(
          std::min(1.0, static_cast<double>(features[item].frames) / 100.0));
      results.push_back(std::move(result));
    }
    return results;
  }

  const char* Name() const override { return "onnxruntime-eres2net"; }

 private:
  Ort::Env environment_;
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
};

}  // namespace

std::unique_ptr<EmbeddingEngine> CreateOnnxEngine(
    const std::string& model_path, std::int32_t intra_op_threads) {
  return std::make_unique<OnnxEngine>(model_path, intra_op_threads);
}

}  // namespace cuebee
