#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuebee/protocol.h"

namespace cuebee {

class EmbeddingEngine {
 public:
  virtual ~EmbeddingEngine() = default;
  virtual std::vector<EmbeddingResult> Infer(
      const std::vector<FeatureMatrix>& features) = 0;
  virtual const char* Name() const = 0;
};

std::unique_ptr<EmbeddingEngine> CreateDeterministicEngine();

#if CUEBEE_ENABLE_ONNX
std::unique_ptr<EmbeddingEngine> CreateOnnxEngine(
    const std::string& model_path, std::int32_t intra_op_threads);
#endif

}  // namespace cuebee
