#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "cuebee/embedding_engine.h"
#include "cuebee/protocol.h"

namespace {

struct Options {
  std::string backend = "deterministic";
  std::string model_path;
  std::int32_t intra_op_threads = 0;
  std::uint32_t max_batch_size = 64;
  std::uint32_t max_frames = 2'000;
};

std::uint32_t ParsePositive(const std::string& name, const std::string& value) {
  std::size_t consumed = 0;
  const unsigned long parsed = std::stoul(value, &consumed);
  if (consumed != value.size() || parsed == 0 || parsed > UINT32_MAX) {
    throw std::invalid_argument("invalid " + name + ": " + value);
  }
  return static_cast<std::uint32_t>(parsed);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      std::cout << "Usage: cuebee-speaker-worker [--backend deterministic|onnx] "
                   "[--model PATH] [--intra-op-threads N] "
                   "[--max-batch-size N] [--max-frames N]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--backend") {
      options.backend = value;
    } else if (argument == "--model") {
      options.model_path = value;
    } else if (argument == "--intra-op-threads") {
      options.intra_op_threads =
          static_cast<std::int32_t>(ParsePositive(argument, value));
    } else if (argument == "--max-batch-size") {
      options.max_batch_size = ParsePositive(argument, value);
    } else if (argument == "--max-frames") {
      options.max_frames = ParsePositive(argument, value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  return options;
}

std::unique_ptr<cuebee::EmbeddingEngine> CreateEngine(const Options& options) {
  if (options.backend == "deterministic") {
    return cuebee::CreateDeterministicEngine();
  }
  if (options.backend == "onnx") {
    if (options.model_path.empty()) {
      throw std::invalid_argument("--model is required for the ONNX backend");
    }
#if CUEBEE_ENABLE_ONNX
    return cuebee::CreateOnnxEngine(options.model_path, options.intra_op_threads);
#else
    throw std::invalid_argument(
        "worker was built without ONNX Runtime; configure CUEBEE_ENABLE_ONNX=ON");
#endif
  }
  throw std::invalid_argument("unsupported backend: " + options.backend);
}

bool Send(const cuebee::Response& response) {
  std::string error;
  if (!cuebee::WriteResponse(std::cout, response, &error)) {
    std::cerr << "response write failed: " << error << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);
#endif
  try {
    const Options options = ParseOptions(argc, argv);
    std::unique_ptr<cuebee::EmbeddingEngine> engine = CreateEngine(options);
    std::cerr << "CueBee native worker ready; backend=" << engine->Name() << '\n';

    while (true) {
      cuebee::Request request;
      std::string read_error;
      const cuebee::ReadOutcome outcome = cuebee::ReadRequest(
          std::cin, options.max_batch_size, options.max_frames, &request, &read_error);
      if (outcome == cuebee::ReadOutcome::kEndOfStream) {
        return 0;
      }
      if (outcome == cuebee::ReadOutcome::kError) {
        cuebee::Response response;
        response.status = cuebee::Status::kInvalidRequest;
        response.request_id = request.request_id;
        response.error = read_error;
        Send(response);
        return 2;
      }

      cuebee::Response response;
      response.request_id = request.request_id;
      if (request.operation == cuebee::Operation::kPing) {
        if (!Send(response)) {
          return 3;
        }
        continue;
      }
      if (request.operation == cuebee::Operation::kShutdown) {
        Send(response);
        return 0;
      }
      if (request.operation != cuebee::Operation::kInfer) {
        response.status = cuebee::Status::kUnsupported;
        response.error = "unsupported worker operation";
      } else {
        try {
          response.results = engine->Infer(request.features);
        } catch (const std::exception& error) {
          response.status = cuebee::Status::kBackendError;
          response.error = error.what();
        }
      }
      if (!Send(response)) {
        return 3;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "worker startup failed: " << error.what() << '\n';
    return 1;
  }
}
