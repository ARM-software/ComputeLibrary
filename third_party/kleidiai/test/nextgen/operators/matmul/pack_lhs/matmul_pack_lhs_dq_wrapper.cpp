//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_dq_wrapper.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

std::string_view MatMulPackLhsDqWrapper::name() const {
    return m_name;
}

size_t MatMulPackLhsDqWrapper::src_tensor_id() const {
    const size_t tensor_id = *m_src_format == PlainFormat(DataType::FP32) ? MATMUL_SLOT_LHS_RAW : MATMUL_SLOT_LHS_DATA;
    return tensor_id;
}

std::vector<size_t> MatMulPackLhsDqWrapper::run_inputs([[maybe_unused]] Span<const Tensor> tensors) const {
    const size_t src_id = src_tensor_id();
    return {src_id};
}

std::vector<size_t> MatMulPackLhsDqWrapper::ref_inputs([[maybe_unused]] Span<const Tensor> tensors) const {
    return {MATMUL_SLOT_LHS_QDATA, MATMUL_SLOT_LHS_QSCALE, MATMUL_SLOT_LHS_QZP_NEG};
}

std::vector<size_t> MatMulPackLhsDqWrapper::steps(Span<const size_t> shape, Span<const Tensor> tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only M and K dimensions are expected.");

    const auto& pack_args = tensors.at(MATMUL_SLOT_PACK_ARGS).value<MatMulPackArgs>();

    const size_t m_step = m_kernel.get_m_step(pack_args.mr);
    const size_t shape_k = shape.at(1);

    return {m_step, shape_k};
}

void MatMulPackLhsDqWrapper::populate_constant_info(Span<Tensor> tensors) const {
    Tensor& lhs_raw = tensors.at(MATMUL_SLOT_LHS_RAW);
    Tensor& packed_lhs = tensors.at(MATMUL_SLOT_IMP_LHS_PACKED);

    lhs_raw.set_format(m_src_format);
    packed_lhs.set_format(m_dst_format);
}

void MatMulPackLhsDqWrapper::run(
    Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    Span<Tensor> tensors) const {
    KAI_TEST_ASSERT_MSG(full_shape.size() == 2, "Only M and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_coords.size() == 2, "Only M and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == 2, "Only M and K dimensions are expected.");

    const size_t full_m = full_shape.at(0);
    const size_t full_k = full_shape.at(1);

    const size_t start_m = tile_coords.at(0);
    const size_t start_k = tile_coords.at(1);

    const size_t size_m = tile_shape.at(0);
    const size_t size_k = tile_shape.at(1);

    KAI_TEST_ASSERT(start_k == 0);
    KAI_TEST_ASSERT(size_k == full_k);

    const size_t lhs_tensor_id = src_tensor_id();
    const Tensor& lhs_data = tensors.at(lhs_tensor_id);
    Tensor& packed_lhs = tensors.at(MATMUL_SLOT_IMP_LHS_PACKED);

    const auto& pack_args = tensors.at(MATMUL_SLOT_PACK_ARGS).value<MatMulPackArgs>();

    packed_lhs.set_shape({full_m, full_k}).allocate();

    const size_t lhs_stride = m_src_format->compute_size({1, full_k});

    const size_t lhs_offset = m_src_format->compute_offset(full_shape, tile_coords);
    const size_t imp_lhs_offset = m_kernel.get_lhs_offset(start_m, lhs_stride);
    KAI_TEST_ASSERT(imp_lhs_offset == lhs_offset);

    const size_t packed_lhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const size_t imp_packed_lhs_offset =
        m_kernel.get_lhs_packed_offset(start_m, full_k, pack_args.mr, pack_args.kr, pack_args.sr);
    KAI_TEST_ASSERT(imp_packed_lhs_offset == packed_lhs_offset);

    const size_t packed_lhs_size = packed_lhs.data().size();
    const size_t imp_packed_lhs_size =
        m_kernel.get_lhs_packed_size(full_m, full_k, pack_args.mr, pack_args.kr, pack_args.sr);
    KAI_TEST_ASSERT(imp_packed_lhs_size == packed_lhs_size);

    const Span<const std::byte> lhs_tile = lhs_data.data().subspan(lhs_offset);
    const Span<std::byte> packed_lhs_tile = packed_lhs.data().subspan(packed_lhs_offset);

    abi_check([&] {
        m_kernel.run(
            size_m, size_k, pack_args.mr, pack_args.kr, pack_args.sr, 0,
            reinterpret_cast<const float*>(lhs_tile.data()), lhs_stride, packed_lhs_tile.data());
    });
}

void MatMulPackLhsDqWrapper::compute_reference(Span<const size_t> shape, Span<Tensor> tensors) const {
    const Tensor& lhs_qdata = tensors.at(MATMUL_SLOT_LHS_QDATA);
    const Tensor& lhs_qscale = tensors.at(MATMUL_SLOT_LHS_QSCALE);
    const Tensor& lhs_qzp_neg = tensors.at(MATMUL_SLOT_LHS_QZP_NEG);
    Tensor& ref_packed_lhs = tensors.at(MATMUL_SLOT_REF_LHS_PACKED);

    ref_packed_lhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(shape, std::array{lhs_qdata.data(), lhs_qzp_neg.data(), lhs_qscale.data()}));
}

}  // namespace kai::test
