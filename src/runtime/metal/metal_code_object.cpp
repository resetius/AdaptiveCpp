/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/runtime/metal/metal_code_object.hpp"

#include <Metal/Metal.hpp>
#include <string_view>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <cstdio>

#undef nil

#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

namespace {

bool parse_requires_atomic64_locks(const std::string& source) {
  constexpr std::string_view prefix = "// capabilities:";
  std::string_view first_line{source.data(), source.find('\n')};
  if (first_line.substr(0, prefix.size()) != prefix) {
    return false;
  }

  auto capabilities = first_line.substr(prefix.size());
  while (!capabilities.empty()) {
    const auto comma = capabilities.find(',');
    auto capability = capabilities.substr(0, comma);
    while (!capability.empty() && (capability.front() == ' ' || capability.front() == '\t'))
      capability.remove_prefix(1);
    while (!capability.empty() && (capability.back() == ' ' || capability.back() == '\t' || capability.back() == '\r'))
      capability.remove_suffix(1);
    if (capability == "atomic64")
      return true;
    if (comma == std::string_view::npos)
      break;
    capabilities.remove_prefix(comma + 1);
  }
  return false;
}

// TEMPORARY: with ACPP_METAL_DUMP_DIRECTORY set, every generated shader is
// written there, so that a source a device refuses can be taken off a CI runner.
void dump_metal_source(const std::string& source,
                       const std::vector<std::string>& kernel_names) {
  const char* dir = std::getenv("ACPP_METAL_DUMP_DIRECTORY");
  if (!dir) {
    return;
  }
  std::string name = kernel_names.empty() ? std::string{"kernel"} : kernel_names.front();
  for (char& c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      c = '_';
    }
  }
  if (name.size() > 180) {
    name.resize(180);
  }
  // a time stamp, so that the order of compilation is visible even though the
  // tests tear the runtime down between cases and reload the backend
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  char prefix[32];
  std::snprintf(prefix, sizeof(prefix), "%lld_", (long long)micros);
  std::string path = std::string{dir} + "/" + prefix + name + ".metal";
  std::ofstream out{path};
  out << source;
  HIPSYCL_DEBUG_INFO << "metal_code_object: dumped shader to " << path << std::endl;
}

result build_metal_library_from_source(MTL::Library*& library,
                                       MTL::Device* device,
                                       const std::string& source,
                                       const std::vector<std::string>& kernel_names) {
  dump_metal_source(source, kernel_names);
  if (!device) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Device is null"});
  }

  NS::Error* error = nullptr;

  NS::SharedPtr<MTL::CompileOptions> options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  options->setLanguageVersion(MTL::LanguageVersion4_0);
  options->setOptimizationLevel(MTL::LibraryOptimizationLevel::LibraryOptimizationLevelSize);

  NS::String* sourceString = NS::String::string(source.c_str(),
                                                NS::UTF8StringEncoding);

  library = device->newLibrary(sourceString, options.get(), &error);

  if (error) {
    std::string error_msg = "metal_code_object: Shader compilation failed";
    if (error->localizedDescription()) {
      error_msg += ": ";
      error_msg += error->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{error_msg});
  }

  if (!library) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Library creation failed "
                                 "without error message"});
  }

  HIPSYCL_DEBUG_INFO << "metal_code_object: Successfully compiled Metal shader"
                     << std::endl;

  return make_success();
}

} // anonymous namespace

metal_sscp_executable_object::metal_sscp_executable_object(
    const std::string &metal_source, const std::string &target_arch,
    hcf_object_id hcf_source, const std::vector<std::string> &kernel_names,
    MTL::Device* device, const kernel_configuration &config)
    : _target_arch{target_arch}, _hcf{hcf_source}, _kernel_names{kernel_names},
      _id{config.generate_id()}, _device{device}, _library{nullptr},
      _msl_source{metal_source},
      _requires_atomic64_locks{parse_requires_atomic64_locks(metal_source)} {
  _build_result = build(metal_source);
}

metal_sscp_executable_object::~metal_sscp_executable_object() {
  if (_library) {
    _library->release();
  }
}

result metal_sscp_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state metal_sscp_executable_object::state() const {
  return _library ? code_object_state::executable : code_object_state::invalid;
}

code_format metal_sscp_executable_object::format() const {
  // Metal uses its own shading language format
  return code_format::native_isa;
}

backend_id metal_sscp_executable_object::managing_backend() const {
  return backend_id::metal;
}

hcf_object_id metal_sscp_executable_object::hcf_source() const {
  return _hcf;
}

std::string metal_sscp_executable_object::target_arch() const {
  return _target_arch;
}

compilation_flow metal_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

std::vector<std::string>
metal_sscp_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

MTL::Library* metal_sscp_executable_object::get_library() const {
  return _library;
}

bool metal_sscp_executable_object::requires_atomic64_locks() const {
  return _requires_atomic64_locks;
}

MTL::Device* metal_sscp_executable_object::get_device() const {
  return _device;
}

result metal_sscp_executable_object::build(const std::string& source) {
  if (_library != nullptr)
    return make_success();

  return build_metal_library_from_source(_library, _device, source, _kernel_names);
}

bool metal_sscp_executable_object::contains(
    const std::string &backend_kernel_name) const {
  for (const auto& kernel_name : _kernel_names) {
    if (kernel_name == backend_kernel_name)
      return true;
  }
  return false;
}

} // namespace rt
} // namespace hipsycl
