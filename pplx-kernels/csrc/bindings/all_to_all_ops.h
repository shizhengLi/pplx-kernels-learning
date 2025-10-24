#pragma once

#include <torch/library.h>

namespace pplx {
void register_all_to_all_ops(torch::Library &m);
} // namespace pplx
