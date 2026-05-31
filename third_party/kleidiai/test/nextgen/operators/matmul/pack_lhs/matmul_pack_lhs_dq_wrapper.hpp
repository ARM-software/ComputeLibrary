//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_interface.hpp"

namespace kai::test {

/// Wrapper for LHS packing kernel with dynamic quantization.
class MatMulPackLhsDqWrapper final : public KernelWrapper {
public:
    /// Creates a new wrapper.
    ///
    /// @param[in] name The kernel name.
    /// @param[in] kernel The kernel interface.
    /// @param[in] src_format The input data format.
    /// @param[in] dst_format The output data format.
    MatMulPackLhsDqWrapper(
        std::string_view name, const MatMulPackLhsDqInterface& kernel, Poly<Format>&& src_format,
        Poly<Format>&& dst_format) :
        m_name(name), m_kernel(kernel), m_src_format(std::move(src_format)), m_dst_format(std::move(dst_format)) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<size_t> run_inputs(Span<const Tensor> tensors) const override;
    [[nodiscard]] std::vector<size_t> ref_inputs(Span<const Tensor> tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(Span<const size_t> shape, Span<const Tensor> tensors) const override;
    void populate_constant_info(Span<Tensor> tensors) const override;
    void run(
        Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
        Span<Tensor> tensors) const override;
    void compute_reference(Span<const size_t> shape, Span<Tensor> tensors) const override;

private:
    [[nodiscard]] size_t src_tensor_id() const;  ///< Determines the tensor ID containing the input data.

    std::string m_name;
    MatMulPackLhsDqInterface m_kernel;
    Poly<Format> m_src_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
