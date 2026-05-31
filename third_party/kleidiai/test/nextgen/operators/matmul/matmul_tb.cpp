//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_tb.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/quantization/quantizer.hpp"
#include "test/nextgen/reference/binary_elementwise.hpp"
#include "test/nextgen/reference/clamp.hpp"
#include "test/nextgen/reference/matmul.hpp"
#include "test/nextgen/reference/reduce.hpp"
#include "test/nextgen/reference/unary_elementwise.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

MatMulTb::MatMulTb(
    size_t shape_m, size_t shape_n, size_t shape_k, MatMulBiasMode bias_mode, float clamp_ratio,
    const MatMulOperator* op) :
    m_shape_m(shape_m),
    m_shape_n(shape_n),
    m_shape_k(shape_k),
    m_bias_mode(bias_mode),
    m_clamp_ratio(clamp_ratio),
    m_op(op),
    m_tensors_required() {
    std::fill(m_tensors_required.begin(), m_tensors_required.end(), false);
}

void MatMulTb::generate_test_data(Rng& rng) {
    populate_config();
    determine_required_tensors();

    // Populates the constant information.
    m_op->matmul->populate_constant_info(m_tensors);

    if (m_op->pack_lhs.has_value()) {
        const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();
        pack_lhs.populate_constant_info(m_tensors);
    }

    if (m_op->pack_rhs.has_value()) {
        const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();
        pack_rhs.populate_constant_info(m_tensors);
    }

    // Generates the raw test data.
    generate_lhs_raw(rng);
    generate_rhs_raw(rng);
    generate_bias_raw(rng);

    compute_rhs_t_raw();  // The transposed RHS data is always needed for reference packing.

    // Quantizes the input data.
    if (m_op->lhs_quant.has_value()) {
        quantize_lhs();
    }

    if (m_op->rhs_quant.has_value()) {
        quantize_rhs_t();
    }

    if (m_op->bias_quant.has_value()) {
        quantize_bias();
    }

    if (m_tensors_required.at(MATMUL_SLOT_LHS_QZP_NEG)) {
        compute_lhs_qzp_neg();
    }

    if (m_tensors_required.at(MATMUL_SLOT_RHS_T_QDATA_SIGN)) {
        compute_rhs_t_qdata_sign();
    }

    if (m_tensors_required.at(MATMUL_SLOT_RHS_T_QDATA_SIGN_SUM)) {
        compute_rhs_t_qdata_sign_sum();
    }

    // Generates reference output.
    if (m_op->pack_lhs.has_value()) {
        compute_ref_packed_lhs();
    }

    if (m_op->pack_rhs.has_value()) {
        compute_ref_packed_rhs();
    }

    compute_ref_matmul();
}

void MatMulTb::populate_config() {
    m_tensors.at(MATMUL_SLOT_CONFIG).set_value(MatMulConfig{m_bias_mode});
}

void MatMulTb::determine_required_tensors() {
    std::vector<const KernelWrapper*> kernels{m_op->matmul.get()};

    if (m_op->pack_lhs.has_value()) {
        kernels.emplace_back(m_op->pack_lhs.value().get());
    }

    if (m_op->pack_rhs.has_value()) {
        kernels.emplace_back(m_op->pack_rhs.value().get());
    }

    for (const KernelWrapper* kernel : kernels) {
        if (kernel != nullptr) {
            const std::vector<size_t> run_inputs = kernel->run_inputs(m_tensors);
            const std::vector<size_t> ref_inputs = kernel->ref_inputs(m_tensors);

            for (const size_t id : run_inputs) {
                m_tensors_required.at(id) = true;
            }

            for (const size_t id : ref_inputs) {
                m_tensors_required.at(id) = true;
            }
        }
    }
}

void MatMulTb::generate_lhs_raw(Rng& rng) {
    const std::array shape{m_shape_m, m_shape_k};
    const Poly<Format> format(std::in_place_type<PlainFormat>, DataType::FP32);
    Tensor& tensor = m_tensors.at(MATMUL_SLOT_LHS_RAW);

    tensor.set_shape(shape).set_format(format).set_data(format->generate_random(shape, rng));
}

void MatMulTb::generate_rhs_raw(Rng& rng) {
    const std::array shape{m_shape_k, m_shape_n};
    const Poly<Format> format(std::in_place_type<PlainFormat>, DataType::FP32);
    Tensor& tensor = m_tensors.at(MATMUL_SLOT_RHS_RAW);

    tensor.set_shape(shape).set_format(format).set_data(format->generate_random(shape, rng));
}

void MatMulTb::generate_bias_raw(Rng& rng) {
    const std::array shape{m_shape_n};
    const Poly<Format> format(std::in_place_type<PlainFormat>, DataType::FP32);
    Tensor& tensor = m_tensors.at(MATMUL_SLOT_BIAS_RAW);

    tensor.set_shape(shape).set_format(format).set_data(format->generate_random(shape, rng));
}

void MatMulTb::compute_rhs_t_raw() {
    const std::array shape{m_shape_n, m_shape_k};
    const Poly<Format> format(std::in_place_type<PlainFormat>, DataType::FP32);
    Tensor& rhs_t_raw = m_tensors.at(MATMUL_SLOT_RHS_T_RAW);
    const Tensor& rhs_raw = m_tensors.at(MATMUL_SLOT_RHS_RAW);

    rhs_t_raw.set_shape(shape).set_format(format).set_data(transpose<float>(rhs_raw.data_ptr(), m_shape_k, m_shape_n));
}

void MatMulTb::quantize_lhs() {
    const Quantizer& lhs_quant = *m_op->lhs_quant.value();

    const std::array lhs_shape{m_shape_m, m_shape_k};
    const Tensor& lhs_raw = m_tensors.at(MATMUL_SLOT_LHS_RAW);
    Tensor& lhs_qdata = m_tensors.at(MATMUL_SLOT_LHS_QDATA);
    Tensor& lhs_qscale = m_tensors.at(MATMUL_SLOT_LHS_QSCALE);
    Tensor& lhs_qzp = m_tensors.at(MATMUL_SLOT_LHS_QZP);

    lhs_quant.dynamic_quantize(DataType::FP32, lhs_shape, lhs_raw.data(), lhs_qdata, lhs_qscale, lhs_qzp);
}

void MatMulTb::quantize_rhs_t() {
    const Quantizer& rhs_quant = *m_op->rhs_quant.value();

    const std::array rhs_t_shape{m_shape_n, m_shape_k};
    const Tensor& rhs_t_raw = m_tensors.at(MATMUL_SLOT_RHS_T_RAW);
    Tensor& rhs_t_qdata = m_tensors.at(MATMUL_SLOT_RHS_T_QDATA);
    Tensor& rhs_t_qscale = m_tensors.at(MATMUL_SLOT_RHS_T_QSCALE);
    Tensor& rhs_t_qzp = m_tensors.at(MATMUL_SLOT_RHS_T_QZP);

    rhs_quant.dynamic_quantize(DataType::FP32, rhs_t_shape, rhs_t_raw.data(), rhs_t_qdata, rhs_t_qscale, rhs_t_qzp);
}

void MatMulTb::quantize_bias() {
    KAI_TEST_ERROR("Not supported.");
}

void MatMulTb::compute_lhs_qzp_neg() {
    const Tensor& lhs_qzp = m_tensors.at(MATMUL_SLOT_LHS_QZP);
    Tensor& lhs_qzp_neg = m_tensors.at(MATMUL_SLOT_LHS_QZP_NEG);

    const Span<const size_t> shape = lhs_qzp.shape();
    const Poly<Format>& format = lhs_qzp.format();

    const UnaryElementwiseFn fn = make_negate(format->dtype());
    Buffer data = fn(shape, lhs_qzp.data());

    lhs_qzp_neg.set_shape(shape).set_format(format).set_data(std::move(data));
}

void MatMulTb::compute_rhs_t_qdata_sign() {
    const Tensor& rhs_t_qdata = m_tensors.at(MATMUL_SLOT_RHS_T_QDATA);
    Tensor& rhs_t_qdata_sign = m_tensors.at(MATMUL_SLOT_RHS_T_QDATA_SIGN);

    const Span<const size_t> shape = rhs_t_qdata.shape();
    const Poly<Format>& format = rhs_t_qdata.format();

    const UnaryElementwiseFn fn = make_change_signedness(format->dtype());
    Buffer data = fn(shape, rhs_t_qdata.data());

    rhs_t_qdata_sign.set_shape(shape).set_format(format).set_data(std::move(data));
}

void MatMulTb::compute_rhs_t_qdata_sign_sum() {
    const Tensor& rhs_t_qdata_sign = m_tensors.at(MATMUL_SLOT_RHS_T_QDATA_SIGN);
    Tensor& rhs_t_qdata_sign_sum = m_tensors.at(MATMUL_SLOT_RHS_T_QDATA_SIGN_SUM);

    const std::array rhs_t_shape = {m_shape_n, m_shape_k};
    const std::array rhs_t_rowsum_shape = {m_shape_n};
    const DataType src_dtype = rhs_t_qdata_sign.format()->dtype();
    const DataType dst_dtype = rhs_t_qdata_sign_sum.format()->dtype();

    const ReduceFn fn = make_reduce_add(src_dtype, dst_dtype);
    Buffer data = fn(0, rhs_t_shape, rhs_t_qdata_sign.data());

    rhs_t_qdata_sign_sum.set_shape(rhs_t_rowsum_shape).set_data(std::move(data));
}

void MatMulTb::compute_ref_packed_lhs() {
    const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();
    const std::array lhs_shape{m_shape_m, m_shape_k};
    pack_lhs.compute_reference(lhs_shape, m_tensors);
}

void MatMulTb::compute_ref_packed_rhs() {
    const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();
    const std::array rhs_t_shape{m_shape_n, m_shape_k};
    pack_rhs.compute_reference(rhs_t_shape, m_tensors);
}

void MatMulTb::compute_ref_matmul() {
    const MatMulConfig& config = m_tensors.at(MATMUL_SLOT_CONFIG).value<MatMulConfig>();
    const Tensor& lhs_qdata = m_tensors.at(MATMUL_SLOT_LHS_QDATA);
    const Tensor& lhs_qscale = m_tensors.at(MATMUL_SLOT_LHS_QSCALE);
    const Tensor& lhs_qzp = m_tensors.at(MATMUL_SLOT_LHS_QZP);
    const Tensor& rhs_t_qdata = m_tensors.at(MATMUL_SLOT_RHS_T_QDATA);
    const Tensor& rhs_t_qscale = m_tensors.at(MATMUL_SLOT_RHS_T_QSCALE);
    const Tensor& bias_raw = m_tensors.at(MATMUL_SLOT_BIAS_RAW);
    Tensor& kernel_args = m_tensors.at(MATMUL_SLOT_MATMUL_ARGS);
    Tensor& ref_dst_data = m_tensors.at(MATMUL_SLOT_REF_DST_DATA);

    ref_dst_data.set_shape({m_shape_m, m_shape_n}).set_format(make_poly<PlainFormat>(m_op->dst_dtype));

    // REVISIT: Assumes that the LHS and RHS are both quantized.
    const Quantizer& lhs_quant = *m_op->lhs_quant.value();
    const Quantizer& rhs_quant = *m_op->rhs_quant.value();

    const Buffer lhs_data = lhs_quant.dequantize(
        m_op->acc_dtype, {m_shape_m, m_shape_k}, lhs_qdata.data(), lhs_qscale.data(), lhs_qzp.data());
    const Buffer rhs_t_data =
        rhs_quant.dequantize(m_op->acc_dtype, {m_shape_n, m_shape_k}, rhs_t_qdata.data(), rhs_t_qscale.data(), {});

    const MatMulFn matmul_fn = make_matmul_nt_t(m_op->acc_dtype);
    Buffer dst = matmul_fn(m_shape_m, m_shape_n, m_shape_k, lhs_data, rhs_t_data);

    switch (config.bias_mode) {
        case MatMulBiasMode::NO_BIAS:
            break;

        case MatMulBiasMode::PER_N: {
            const BinaryElementwiseFn add_fn = make_add_2d(m_op->acc_dtype);
            dst = add_fn(m_shape_m, m_shape_n, dst, 1, m_shape_n, bias_raw.data());
            break;
        }

        default:
            KAI_TEST_ERROR("Not supported.");
    }

    const DynamicClampFn dynamic_clamp_fn = make_dynamic_clamp(m_op->acc_dtype);
    auto [clamp_args, clampped_dst] = dynamic_clamp_fn(m_clamp_ratio, {m_shape_m, m_shape_n}, dst);

    kernel_args.set_shape({clamp_args.size()}).set_data(std::move(clamp_args));

    KAI_TEST_ASSERT_MSG(
        m_op->dst_dtype == m_op->acc_dtype, "Only support the accumulator and output type being the same.");
    ref_dst_data.set_data(std::move(clampped_dst));
}

bool MatMulTb::has_lhs_packing() const {
    return m_op->pack_lhs != nullptr;
}

std::tuple<size_t, size_t> MatMulTb::lhs_packing_steps() const {
    const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();
    const std::vector<size_t> steps = pack_lhs.steps({m_shape_m, m_shape_k}, m_tensors);
    return {steps.at(0), steps.at(1)};
}

void MatMulTb::test_lhs_packing(size_t start_m, size_t start_k, size_t size_m, size_t size_k) {
    const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();

    const std::array full_shape{m_shape_m, m_shape_k};
    const std::array tile_coords{start_m, start_k};
    const std::array tile_shape{size_m, size_k};

    pack_lhs.run(full_shape, tile_coords, tile_shape, m_tensors);

    const Tensor& ref_packed_lhs = m_tensors.at(MATMUL_SLOT_REF_LHS_PACKED);
    const Tensor& imp_packed_lhs = m_tensors.at(MATMUL_SLOT_IMP_LHS_PACKED);
    const Format& format = *ref_packed_lhs.format();

    DefaultMismatchHandler handler(0.0F, 0.0F, 0, 0.0F);
    const bool ok =
        format.compare(full_shape, tile_coords, tile_shape, imp_packed_lhs.data(), ref_packed_lhs.data(), handler);
    KAI_TEST_ASSERT(ok);
}

bool MatMulTb::has_rhs_packing() const {
    return m_op->pack_rhs.has_value();
}

std::tuple<size_t, size_t> MatMulTb::rhs_packing_steps() const {
    const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();
    const std::vector<size_t> steps = pack_rhs.steps({m_shape_n, m_shape_k}, m_tensors);
    return {steps.at(0), steps.at(1)};
}

void MatMulTb::test_rhs_packing(size_t start_n, size_t start_k, size_t size_n, size_t size_k) {
    const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();

    const std::array full_shape{m_shape_n, m_shape_k};
    const std::array tile_coords{start_n, start_k};
    const std::array tile_shape{size_n, size_k};

    pack_rhs.run(full_shape, tile_coords, tile_shape, m_tensors);

    const Tensor& ref_packed_rhs = m_tensors.at(MATMUL_SLOT_REF_RHS_PACKED);
    const Tensor& imp_packed_rhs = m_tensors.at(MATMUL_SLOT_IMP_RHS_PACKED);
    const Format& format = *ref_packed_rhs.format();

    DefaultMismatchHandler handler(0.0F, 0.0F, 0, 0.0F);
    const bool ok =
        format.compare(full_shape, tile_coords, tile_shape, imp_packed_rhs.data(), ref_packed_rhs.data(), handler);
    KAI_TEST_ASSERT(ok);
}

std::tuple<size_t, size_t> MatMulTb::matmul_steps() const {
    const std::vector<size_t> steps = m_op->matmul->steps({m_shape_m, m_shape_n, m_shape_k}, m_tensors);
    return {steps.at(0), steps.at(1)};
}

void MatMulTb::test_matmul(size_t start_m, size_t start_n, size_t size_m, size_t size_n) {
    const std::array matmul_full_shape{m_shape_m, m_shape_n, m_shape_k};
    const std::array matmul_tile_coords{start_m, start_n, static_cast<size_t>(0)};
    const std::array matmul_tile_shape{size_m, size_n, m_shape_k};

    const std::array dst_full_shape{m_shape_m, m_shape_n};
    const std::array dst_tile_coords{start_m, start_n};
    const std::array dst_tile_shape{size_m, size_n};

    m_op->matmul->run(matmul_full_shape, matmul_tile_coords, matmul_tile_shape, m_tensors);

    const Tensor& ref_dst_data = m_tensors.at(MATMUL_SLOT_REF_DST_DATA);
    const Tensor& imp_dst_data = m_tensors.at(MATMUL_SLOT_IMP_DST_DATA);
    const Format& format = *ref_dst_data.format();

    DefaultMismatchHandler handler(1e-3, 1e-3, 0, 0.0F);
    const bool ok = format.compare(
        dst_full_shape, dst_tile_coords, dst_tile_shape, imp_dst_data.data(), ref_dst_data.data(), handler);
    KAI_TEST_ASSERT(ok);
}

}  // namespace kai::test
