// Copyright © 2023 Apple Inc.

#include <cstdlib>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/scheduler.h"

namespace mlx::core::allocator {

Buffer malloc(size_t size) {
  auto buffer = allocator().malloc(size);
  if (size && !buffer.ptr()) {
    std::ostringstream msg;
    msg << "[malloc] Unable to allocate " << size << " bytes.";
    throw std::runtime_error(msg.str());
  }
  return buffer;
}

void free(Buffer buffer) {
  allocator().free(buffer);
}

Buffer CommonAllocator::malloc(size_t size) {
  void* ptr = std::malloc(size + sizeof(size_t));
  if (ptr != nullptr) {
    *static_cast<size_t*>(ptr) = size;
  }
  return Buffer{ptr};
}

void CommonAllocator::free(Buffer buffer) {
  std::free(buffer.ptr());
}

size_t CommonAllocator::size(Buffer buffer) const {
  if (buffer.ptr() == nullptr) {
    return 0;
  }
  return *static_cast<size_t*>(buffer.ptr());
}

} // namespace mlx::core::allocator
