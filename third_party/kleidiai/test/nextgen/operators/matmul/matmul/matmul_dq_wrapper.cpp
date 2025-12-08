//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul/matmul_dq_wrapper.hpp"

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_main_args.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/reference/binary_elementwise.hpp"
#include "test/nextgen/reference/clamp.hpp"
#include "test/nextgen/reference/matmul.hpp"

namespace kai::test {

std::string_view MatMulDqWrapper::name() const {
    return m_name;
}

std::vector<size_t> MatMulDqWrapper::run_inputs([[maybe_unused]] Span<const Tensor> tensors) const {
    return {MATMUL_SLOT_REF_LHS_PACKED, MATMUL_SLOT_REF_RHS_PACKED, MATMUL_SLOT_MATMUL_ARGS};
}

std::vector<size_t> MatMulDqWrapper::ref_inputs([[maybe_unused]] Span<const Tensor> tensors) const {
    return {MATMUL_SLOT_LHS_QDATA,   MATMUL_SLOT_LHS_QSCALE,   MATMUL_SLOT_LHS_QZP,
            MATMUL_SLOT_RHS_T_QDATA, MATMUL_SLOT_RHS_T_QSCALE, MATMUL_SLOT_BIAS_RAW};
}

std::vector<size_t> MatMulDqWrapper::steps(
    Span<const size_t> shape, [[maybe_unused]] Span<const Tensor> tensorsf) const {
    const size_t step_m = m_kernel.get_m_step();
    const size_t step_n = m_kernel.get_n_step();
    const size_t shape_k = shape.at(2);

    return {step_m, step_n, shape_k};
}

void MatMulDqWrapper::populate_constant_info(Span<Tensor> tensors) const {
    // Populates the packing arguments.
    Tensor& pack_args_tensor = tensors.at(MATMUL_SLOT_PACK_ARGS);
    pack_args_tensor.set_shape({sizeof(MatMulPackArgs)}).allocate();
    auto& pack_args = pack_args_tensor.value<MatMulPackArgs>();

    pack_args.mr = m_kernel.get_mr();
    pack_args.nr = m_kernel.get_nr();
    pack_args.kr = m_kernel.get_kr();
    pack_args.sr = m_kernel.get_sr();
    pack_args.bl = 0;

    // Setups data format.
}

void MatMulDqWrapper::run(
    Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    Span<Tensor> tensors) const {
    KAI_TEST_ASSERT(tile_coords.size() == full_shape.size());
    KAI_TEST_ASSERT(tile_shape.size() == full_shape.size());

    KAI_TEST_ASSERT_MSG(full_shape.size() == 3, "Only M, N and K dimensions are expected.");

    const size_t full_m = full_shape.at(0);
    const size_t full_n = full_shape.at(1);
    const size_t full_k = full_shape.at(2);

    const size_t start_m = tile_coords.at(0);
    const size_t start_n = tile_coords.at(1);
    const size_t start_k = tile_coords.at(2);

    const size_t size_m = tile_shape.at(0);
    const size_t size_n = tile_shape.at(1);
    const size_t size_k = tile_shape.at(2);

    KAI_TEST_ASSERT_MSG(start_k == 0, "Only full K is supported.");
    KAI_TEST_ASSERT_MSG(size_k == full_k, "Only full K is supported.");

    const Tensor& ref_packed_lhs = tensors.at(MATMUL_SLOT_REF_LHS_PACKED);
    const Tensor& ref_packed_rhs = tensors.at(MATMUL_SLOT_REF_RHS_PACKED);
    const Tensor& kernel_args = tensors.at(MATMUL_SLOT_MATMUL_ARGS);
    Tensor& imp_dst_data = tensors.at(MATMUL_SLOT_IMP_DST_DATA);

    const size_t ref_packed_lhs_offset = m_lhs_format->compute_offset({full_m, full_k}, {start_m, start_k});
    const size_t imp_packed_lhs_offset = m_kernel.get_lhs_packed_offset(start_m, full_k);
    KAI_TEST_ASSERT(imp_packed_lhs_offset == ref_packed_lhs_offset);

    const size_t ref_packed_rhs_offset = m_rhs_format->compute_offset({full_n, full_k}, {start_n, start_k});
    const size_t imp_packed_rhs_offset = m_kernel.get_rhs_packed_offset(start_n, full_k);
    KAI_TEST_ASSERT(imp_packed_rhs_offset == ref_packed_rhs_offset);

    const size_t ref_dst_stride_row = m_dst_format->compute_size({full_n});
    const size_t ref_dst_stride_col = m_dst_format->compute_size({1});
    const size_t ref_dst_offset = m_dst_format->compute_offset({full_m, full_n}, {start_m, start_n});
    const size_t imp_dst_offset = m_kernel.get_dst_offset(start_m, start_n, ref_dst_stride_row);
    KAI_TEST_ASSERT(imp_dst_offset == ref_dst_offset);

    imp_dst_data.set_shape({full_m, full_n}).set_format(m_dst_format).allocate();
    const size_t imp_dst_size = m_kernel.get_dst_size(full_m, full_n);
    KAI_TEST_ASSERT(imp_dst_size == imp_dst_data.data().size());

    const Span<const std::byte> packed_lhs_tile = ref_packed_lhs.data().subspan(ref_packed_lhs_offset);
    const Span<const std::byte> packed_rhs_tile = ref_packed_rhs.data().subspan(ref_packed_rhs_offset);
    const Span<std::byte> dst_tile = imp_dst_data.data().subspan(ref_dst_offset);

    const auto& clamp_args = kernel_args.value<MatMulClampArgsF32>();

    abi_check([&] {
        m_kernel.run(
            size_m, size_n, size_k, packed_lhs_tile.data(), packed_rhs_tile.data(),
            reinterpret_cast<float*>(dst_tile.data()), ref_dst_stride_row, ref_dst_stride_col, clamp_args.clamp_min,
            clamp_args.clamp_max);
    });
}

void MatMulDqWrapper::compute_reference(
    [[maybe_unused]] Span<const size_t> shape, [[maybe_unused]] Span<Tensor> tensors) const {
}

}  // namespace kai::test
