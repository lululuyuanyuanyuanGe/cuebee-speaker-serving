#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace cuebee {

constexpr std::uint16_t kProtocolVersion = 1;
constexpr std::uint32_t kExpectedMelBins = 80;
constexpr std::uint32_t kEmbeddingDimension = 192;

enum class Operation : std::uint16_t {
  kInfer = 1,
  kPing = 2,
  kShutdown = 3,
};

enum class Status : std::uint16_t {
  kOk = 0,
  kInvalidRequest = 1,
  kBackendError = 2,
  kUnsupported = 3,
};

struct FeatureMatrix {
  std::uint32_t frames = 0;
  std::vector<float> values;
};

struct Request {
  Operation operation = Operation::kPing;
  std::uint64_t request_id = 0;
  std::uint32_t mel_bins = kExpectedMelBins;
  std::vector<FeatureMatrix> features;
};

struct EmbeddingResult {
  float quality = 0.0F;
  std::vector<float> embedding;
};

struct Response {
  Status status = Status::kOk;
  std::uint64_t request_id = 0;
  std::string error;
  std::vector<EmbeddingResult> results;
};

enum class ReadOutcome {
  kOk,
  kEndOfStream,
  kError,
};

ReadOutcome ReadRequest(std::istream& input, std::uint32_t max_batch_size,
                        std::uint32_t max_frames, Request* request,
                        std::string* error);
bool WriteResponse(std::ostream& output, const Response& response,
                   std::string* error);

}  // namespace cuebee
