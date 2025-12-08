//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_interface.hpp"

namespace kai::test {

/// Wrapper for RHS packing kernel with per-channel quantization.
class MatMulPackRhsQuantWrapper : public KernelWrapper {
public:
    /// Creates a new wrapper.
    MatMulPackRhsQuantWrapper(
        std::string_view name, const MatMulPackRhsQuantInterface& kernel, const Poly<Format>& src_data_format,
        const Poly<Format>& src_scale_format, const Poly<Format>& src_bias_format, const Poly<Format>& src_sum_format,
        const Poly<Format>& dst_format) :
        m_name(name),
        m_kernel(kernel),
        m_src_data_format(src_data_format),
        m_src_scale_format(src_scale_format),
        m_src_bias_format(src_bias_format),
        m_src_sum_format(src_sum_format),
        m_dst_format(dst_format) {
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
    std::string m_name;
    MatMulPackRhsQuantInterface m_kernel;
    Poly<Format> m_src_data_format;
    Poly<Format> m_src_scale_format;
    Poly<Format> m_src_bias_format;
    Poly<Format> m_src_sum_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
