#include "cuebee/protocol.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>

namespace cuebee {
namespace {

constexpr std::array<char, 4> kRequestMagic = {'C', 'B', 'S', 'P'};
constexpr std::array<char, 4> kResponseMagic = {'C', 'B', 'S', 'R'};
constexpr std::size_t kHeaderBytes = 32;

bool ReadExact(std::istream& input, char* destination, std::size_t size) {
  input.read(destination, static_cast<std::streamsize>(size));
  return static_cast<std::size_t>(input.gcount()) == size;
}

std::uint16_t DecodeU16(const char* value) {
  return static_cast<std::uint16_t>(static_cast<unsigned char>(value[0])) |
         (static_cast<std::uint16_t>(static_cast<unsigned char>(value[1])) << 8U);
}

std::uint32_t DecodeU32(const char* value) {
  std::uint32_t result = 0;
  for (std::uint32_t index = 0; index < 4; ++index) {
    result |= static_cast<std::uint32_t>(
                  static_cast<unsigned char>(value[index]))
              << (index * 8U);
  }
  return result;
}

std::uint64_t DecodeU64(const char* value) {
  std::uint64_t result = 0;
  for (std::uint32_t index = 0; index < 8; ++index) {
    result |= static_cast<std::uint64_t>(
                  static_cast<unsigned char>(value[index]))
              << (index * 8U);
  }
  return result;
}

float DecodeFloat(const char* value) {
  const std::uint32_t bits = DecodeU32(value);
  float result = 0.0F;
  static_assert(sizeof(result) == sizeof(bits), "float must be 32 bits");
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void EncodeU16(std::uint16_t value, std::ostream& output) {
  for (std::uint32_t index = 0; index < 2; ++index) {
    output.put(static_cast<char>((value >> (index * 8U)) & 0xFFU));
  }
}

void EncodeU32(std::uint32_t value, std::ostream& output) {
  for (std::uint32_t index = 0; index < 4; ++index) {
    output.put(static_cast<char>((value >> (index * 8U)) & 0xFFU));
  }
}

void EncodeU64(std::uint64_t value, std::ostream& output) {
  for (std::uint32_t index = 0; index < 8; ++index) {
    output.put(static_cast<char>((value >> (index * 8U)) & 0xFFU));
  }
}

void EncodeFloat(float value, std::ostream& output) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  EncodeU32(bits, output);
}

bool CheckedMultiply(std::uint64_t left, std::uint64_t right,
                     std::uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

}  // namespace

ReadOutcome ReadRequest(std::istream& input, std::uint32_t max_batch_size,
                        std::uint32_t max_frames, Request* request,
                        std::string* error) {
  std::array<char, kHeaderBytes> header{};
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (input.gcount() == 0 && input.eof()) {
    return ReadOutcome::kEndOfStream;
  }
  if (static_cast<std::size_t>(input.gcount()) != header.size()) {
    *error = "truncated request header";
    return ReadOutcome::kError;
  }

  request->request_id = DecodeU64(header.data() + 8);
  if (!std::equal(kRequestMagic.begin(), kRequestMagic.end(), header.begin())) {
    *error = "invalid request magic";
    return ReadOutcome::kError;
  }
  if (DecodeU16(header.data() + 4) != kProtocolVersion) {
    *error = "unsupported protocol version";
    return ReadOutcome::kError;
  }

  request->operation = static_cast<Operation>(DecodeU16(header.data() + 6));
  const std::uint32_t batch_size = DecodeU32(header.data() + 16);
  request->mel_bins = DecodeU32(header.data() + 20);
  const std::uint32_t payload_bytes = DecodeU32(header.data() + 24);
  if (DecodeU32(header.data() + 28) != 0) {
    *error = "reserved request field must be zero";
    return ReadOutcome::kError;
  }
  if (batch_size > max_batch_size) {
    *error = "batch exceeds configured maximum";
    return ReadOutcome::kError;
  }

  std::uint64_t maximum_values = 0;
  std::uint64_t maximum_bytes = 0;
  if (!CheckedMultiply(max_batch_size, max_frames, &maximum_values) ||
      !CheckedMultiply(maximum_values, kExpectedMelBins, &maximum_values) ||
      !CheckedMultiply(maximum_values, sizeof(float), &maximum_bytes)) {
    *error = "configured request limit overflows";
    return ReadOutcome::kError;
  }
  maximum_bytes += static_cast<std::uint64_t>(max_batch_size) * sizeof(std::uint32_t);
  if (payload_bytes > maximum_bytes) {
    *error = "request payload exceeds configured maximum";
    return ReadOutcome::kError;
  }

  std::vector<char> payload(payload_bytes);
  if (payload_bytes > 0 && !ReadExact(input, payload.data(), payload.size())) {
    *error = "truncated request payload";
    return ReadOutcome::kError;
  }

  if (request->operation != Operation::kInfer) {
    if (batch_size != 0 || payload_bytes != 0) {
      *error = "control request must not contain a payload";
      return ReadOutcome::kError;
    }
    request->features.clear();
    return ReadOutcome::kOk;
  }
  if (batch_size == 0) {
    *error = "inference batch must not be empty";
    return ReadOutcome::kError;
  }
  if (request->mel_bins != kExpectedMelBins) {
    *error = "worker expects 80 log-Mel bins";
    return ReadOutcome::kError;
  }

  request->features.clear();
  request->features.reserve(batch_size);
  std::size_t offset = 0;
  for (std::uint32_t item_index = 0; item_index < batch_size; ++item_index) {
    if (payload.size() - offset < sizeof(std::uint32_t)) {
      *error = "missing feature frame count";
      return ReadOutcome::kError;
    }
    const std::uint32_t frames = DecodeU32(payload.data() + offset);
    offset += sizeof(std::uint32_t);
    if (frames == 0 || frames > max_frames) {
      *error = "invalid feature frame count";
      return ReadOutcome::kError;
    }
    std::uint64_t value_count = 0;
    std::uint64_t byte_count = 0;
    if (!CheckedMultiply(frames, request->mel_bins, &value_count) ||
        !CheckedMultiply(value_count, sizeof(float), &byte_count) ||
        byte_count > payload.size() - offset) {
      *error = "truncated feature matrix";
      return ReadOutcome::kError;
    }
    FeatureMatrix matrix;
    matrix.frames = frames;
    matrix.values.reserve(static_cast<std::size_t>(value_count));
    for (std::uint64_t value_index = 0; value_index < value_count; ++value_index) {
      matrix.values.push_back(
          DecodeFloat(payload.data() + offset + value_index * sizeof(float)));
    }
    offset += static_cast<std::size_t>(byte_count);
    request->features.push_back(std::move(matrix));
  }
  if (offset != payload.size()) {
    *error = "request payload has trailing bytes";
    return ReadOutcome::kError;
  }
  return ReadOutcome::kOk;
}

bool WriteResponse(std::ostream& output, const Response& response,
                   std::string* error) {
  std::ostringstream payload(std::ios::binary);
  payload.write(response.error.data(), static_cast<std::streamsize>(response.error.size()));
  if (response.status == Status::kOk) {
    for (const EmbeddingResult& result : response.results) {
      if (result.embedding.size() != kEmbeddingDimension) {
        *error = "backend returned an invalid embedding dimension";
        return false;
      }
      EncodeFloat(result.quality, payload);
      for (float value : result.embedding) {
        EncodeFloat(value, payload);
      }
    }
  }
  const std::string bytes = payload.str();
  if (bytes.size() > std::numeric_limits<std::uint32_t>::max() ||
      response.error.size() > std::numeric_limits<std::uint32_t>::max() ||
      response.results.size() > std::numeric_limits<std::uint32_t>::max()) {
    *error = "response exceeds protocol limits";
    return false;
  }

  output.write(kResponseMagic.data(), static_cast<std::streamsize>(kResponseMagic.size()));
  EncodeU16(kProtocolVersion, output);
  EncodeU16(static_cast<std::uint16_t>(response.status), output);
  EncodeU64(response.request_id, output);
  EncodeU32(static_cast<std::uint32_t>(response.results.size()), output);
  EncodeU32(response.results.empty() ? 0U : kEmbeddingDimension, output);
  EncodeU32(static_cast<std::uint32_t>(bytes.size()), output);
  EncodeU32(static_cast<std::uint32_t>(response.error.size()), output);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.flush();
  if (!output.good()) {
    *error = "failed to write worker response";
    return false;
  }
  return true;
}

}  // namespace cuebee
