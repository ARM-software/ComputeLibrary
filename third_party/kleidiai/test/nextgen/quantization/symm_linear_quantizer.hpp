//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/functions/round.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/quantization/quantizer.hpp"

namespace kai::test {

/// Symmetric linear quantizer.
class SymmLinearQuantizer : public Quantizer {
public:
    /// Creates a new symmetric linear quantizer.
    ///
    /// @param[in] qdata_dtype The quantized data type.
    /// @param[in] qscale_dtype The quantization scale data type.
    /// @param[in] qdata_round_mode The rounding mode to calculate quantized data.
    /// @param[in] block_height The quantization block height (0 if it's full height).
    /// @param[in] block_width The quantization block width (0 if it's full width).
    SymmLinearQuantizer(
        DataType qdata_dtype, DataType qscale_dtype, RoundMode qdata_round_mode, size_t block_height,
        size_t block_width) :
        m_qdata_dtype(qdata_dtype),
        m_qscale_dtype(qscale_dtype),
        m_qdata_round_mode(qdata_round_mode),
        m_block_height(block_height),
        m_block_width(block_width) {
    }

    void dynamic_quantize(
        DataType fp_dtype, Span<const size_t> shape, Span<const std::byte> fp_data, Tensor& qdata, Tensor& qscale,
        Tensor& qzp) const override;
    [[nodiscard]] Buffer dequantize(
        DataType fp_dtype, Span<const size_t> shape, Span<const std::byte> qdata, Span<const std::byte> qscale,
        Span<const std::byte> qzp) const override;

private:
    DataType m_qdata_dtype;
    DataType m_qscale_dtype;

    RoundMode m_qdata_round_mode;

    size_t m_block_height;
    size_t m_block_width;
};

}  // namespace kai::test
