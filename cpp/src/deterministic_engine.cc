#include "cuebee/embedding_engine.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cuebee {
namespace {

class DeterministicEngine final : public EmbeddingEngine {
 public:
  std::vector<EmbeddingResult> Infer(
      const std::vector<FeatureMatrix>& features) override {
    std::vector<EmbeddingResult> results;
    results.reserve(features.size());
    for (const FeatureMatrix& matrix : features) {
      if (matrix.frames == 0 ||
          matrix.values.size() !=
              static_cast<std::size_t>(matrix.frames) * kExpectedMelBins) {
        throw std::invalid_argument("invalid feature matrix shape");
      }

      std::vector<double> means(kExpectedMelBins, 0.0);
      double square_sum = 0.0;
      for (std::uint32_t frame = 0; frame < matrix.frames; ++frame) {
        for (std::uint32_t mel = 0; mel < kExpectedMelBins; ++mel) {
          const float value = matrix.values[frame * kExpectedMelBins + mel];
          if (!std::isfinite(value)) {
            throw std::invalid_argument("feature matrix contains a non-finite value");
          }
          means[mel] += value;
          square_sum += static_cast<double>(value) * value;
        }
      }
      for (double& mean : means) {
        mean /= matrix.frames;
      }

      EmbeddingResult result;
      result.embedding.resize(kEmbeddingDimension);
      double norm_squared = 0.0;
      for (std::uint32_t index = 0; index < kEmbeddingDimension; ++index) {
        const double mixed = means[index % kExpectedMelBins] * 0.37 +
                             means[(index * 13U + 7U) % kExpectedMelBins] * 0.19 +
                             std::sin(static_cast<double>(index + 1U) * 0.17) * 0.05;
        const float value = static_cast<float>(std::tanh(mixed));
        result.embedding[index] = value;
        norm_squared += static_cast<double>(value) * value;
      }
      const double norm = std::sqrt(norm_squared);
      if (!(norm > 1e-12)) {
        result.embedding[0] = 1.0F;
      } else {
        for (float& value : result.embedding) {
          value = static_cast<float>(value / norm);
        }
      }

      const double count = static_cast<double>(matrix.values.size());
      double mean_square = square_sum / count;
      double mean_value = 0.0;
      for (double mean : means) {
        mean_value += mean;
      }
      mean_value /= kExpectedMelBins;
      const double deviation = std::sqrt(std::max(0.0, mean_square - mean_value * mean_value));
      const double duration = std::min(1.0, matrix.frames / 100.0);
      result.quality = static_cast<float>(duration * std::min(1.0, deviation / 2.0));
      results.push_back(std::move(result));
    }
    return results;
  }

  const char* Name() const override { return "deterministic-development"; }
};

}  // namespace

std::unique_ptr<EmbeddingEngine> CreateDeterministicEngine() {
  return std::make_unique<DeterministicEngine>();
}

}  // namespace cuebee
