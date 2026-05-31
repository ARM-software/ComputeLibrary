//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/quantization/symm_linear_quantizer.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/reference/dequantize.hpp"
#include "test/nextgen/reference/quantize.hpp"

namespace kai::test {

void SymmLinearQuantizer::dynamic_quantize(
    DataType fp_dtype, Span<const size_t> shape, Span<const std::byte> fp_data, Tensor& qdata, Tensor& qscale,
    [[maybe_unused]] Tensor& qzp) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only 2D quantization is supported.");

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t block_height = m_block_height != 0 ? m_block_height : height;
    const size_t block_width = m_block_width != 0 ? m_block_width : width;

    const size_t quant_height = round_up_division(height, block_height);
    const size_t quant_width = round_up_division(width, block_width);
    const std::array quant_shape{quant_height, quant_width};

    const DynamicQuantizeLinearFn quantize_fn =
        make_dynamic_symmetric_quantize_linear(fp_dtype, m_qdata_dtype, m_qscale_dtype, m_qdata_round_mode);
    auto [qdata_buffer, qscale_buffer, qzp_buffer] = quantize_fn(height, width, block_height, block_width, fp_data);

    qdata.set_shape(shape).set_format(make_poly<PlainFormat>(m_qdata_dtype)).set_data(std::move(qdata_buffer));
    qscale.set_shape(quant_shape).set_format(make_poly<PlainFormat>(m_qscale_dtype)).set_data(std::move(qscale_buffer));
}

Buffer SymmLinearQuantizer::dequantize(
    DataType fp_dtype, Span<const size_t> shape, Span<const std::byte> qdata, Span<const std::byte> qscale,
    Span<const std::byte> qzp) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only 2D quantization is supported.");

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t block_height = m_block_height != 0 ? m_block_height : height;
    const size_t block_width = m_block_width != 0 ? m_block_width : width;

    const DequantizeLinearFn fn = make_dequantize_linear(fp_dtype, m_qdata_dtype, m_qscale_dtype);
    Buffer fp_data = fn(height, width, block_height, block_width, qdata, qscale, qzp);

    return fp_data;
}

}  // namespace kai::test
