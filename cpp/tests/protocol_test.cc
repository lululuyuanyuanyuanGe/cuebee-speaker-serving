#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include "cuebee/protocol.h"

namespace {

void AppendU16(std::string* bytes, std::uint16_t value) {
  for (std::uint32_t index = 0; index < 2; ++index) {
    bytes->push_back(static_cast<char>((value >> (index * 8U)) & 0xFFU));
  }
}

void AppendU32(std::string* bytes, std::uint32_t value) {
  for (std::uint32_t index = 0; index < 4; ++index) {
    bytes->push_back(static_cast<char>((value >> (index * 8U)) & 0xFFU));
  }
}

void AppendU64(std::string* bytes, std::uint64_t value) {
  for (std::uint32_t index = 0; index < 8; ++index) {
    bytes->push_back(static_cast<char>((value >> (index * 8U)) & 0xFFU));
  }
}

void AppendFloat(std::string* bytes, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  AppendU32(bytes, bits);
}

}  // namespace

int main() {
  std::string payload;
  AppendU32(&payload, 2);
  for (std::uint32_t index = 0; index < 160; ++index) {
    AppendFloat(&payload, static_cast<float>(index) / 10.0F);
  }

  std::string wire = "CBSP";
  AppendU16(&wire, cuebee::kProtocolVersion);
  AppendU16(&wire, static_cast<std::uint16_t>(cuebee::Operation::kInfer));
  AppendU64(&wire, 42);
  AppendU32(&wire, 1);
  AppendU32(&wire, cuebee::kExpectedMelBins);
  AppendU32(&wire, static_cast<std::uint32_t>(payload.size()));
  AppendU32(&wire, 0);
  wire += payload;

  std::istringstream input(wire, std::ios::binary);
  cuebee::Request request;
  std::string error;
  assert(cuebee::ReadRequest(input, 8, 100, &request, &error) ==
         cuebee::ReadOutcome::kOk);
  assert(request.request_id == 42);
  assert(request.features.size() == 1);
  assert(request.features[0].frames == 2);
  assert(request.features[0].values.size() == 160);
  assert(std::abs(request.features[0].values[159] - 15.9F) < 1e-5F);

  cuebee::Response response;
  response.request_id = 42;
  cuebee::EmbeddingResult result;
  result.quality = 0.75F;
  result.embedding.assign(cuebee::kEmbeddingDimension, 0.5F);
  response.results.push_back(result);
  std::ostringstream output(std::ios::binary);
  assert(cuebee::WriteResponse(output, response, &error));
  const std::string encoded = output.str();
  assert(encoded.size() == 32 + (cuebee::kEmbeddingDimension + 1) * sizeof(float));
  assert(encoded.substr(0, 4) == "CBSR");
  return 0;
}
