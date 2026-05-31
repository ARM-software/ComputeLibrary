//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_interface.hpp"
#include "test/nextgen/quantization/quantizer.hpp"

namespace kai::test {

/// Wrapper for matrix multiplication kernel with dynamic quantization.
class MatMulDqWrapper : public KernelWrapper {
public:
    /// Creates a new wrapper.
    MatMulDqWrapper(
        std::string_view name, const MatMulDqInterface& kernel, std::unique_ptr<Quantizer> lhs_quant,
        std::unique_ptr<Quantizer> rhs_quant, const Poly<Format>& lhs_format, const Poly<Format>& rhs_format,
        const Poly<Format>& dst_format) :
        m_name(name),
        m_kernel(kernel),
        m_lhs_quant(std::move(lhs_quant)),
        m_rhs_quant(std::move(rhs_quant)),
        m_lhs_format(lhs_format),
        m_rhs_format(rhs_format),
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
    MatMulDqInterface m_kernel;
    std::unique_ptr<Quantizer> m_lhs_quant;
    std::unique_ptr<Quantizer> m_rhs_quant;
    Poly<Format> m_lhs_format;
    Poly<Format> m_rhs_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
