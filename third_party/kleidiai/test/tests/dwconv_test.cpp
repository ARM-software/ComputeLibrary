//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/dwconv.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"

namespace kai::test {

namespace {

/// Interface for depthwise kernel.
struct DepthwisePlanarKernel {
    std::function<size_t(size_t m, size_t n, size_t k)> get_dst_size;
    std::function<size_t(size_t m, size_t n)> get_dst_offset;
    std::function<size_t(void)> get_m_step;
    std::function<void(
        const void* inptr, const void* packed_rhs, void* outptr_start, size_t stride_in_row, size_t stride_in_col,
        size_t dst_stride_row, size_t dst_stride_col, unsigned int valid_input_rows, unsigned int valid_out_rows,
        unsigned int pad_left, unsigned int pad_top, float pad_value, float clamp_min, float clamp_max)>
        conv;
};

// Rhs packing micro-kernel.
struct RhsPackDepthwiseKernel {
    std::function<size_t(size_t fh, size_t fw, size_t nc)> get_rhs_packed_size;
    std::function<void(
        size_t filter_height, size_t filter_width, size_t height, size_t width, size_t num_channels, const void* rhs,
        const void* bias, void* rhs_packed)>
        pack;
};

/// Description of a Depthwise kernel set
struct Depthwise {
    std::string_view name;
    std::function<bool(void)> is_supported;
    std::pair<unsigned int, unsigned int> filter;
    DataType data_type;
    DataType acc_type;
    RhsPackDepthwiseKernel rhs;
    DepthwisePlanarKernel depthwise;
};

/// Convenience types for testing.
using DepthwiseArray = std::array<Depthwise, 1>;
using DepthwiseParamsParams = std::tuple<Depthwise, MatMulShape, Padding2D, float>;
using DepthwisePlanarTest = testing::TestWithParam<DepthwiseParamsParams>;

/// Use interface for depthwise kernel
const kai_dwconv_clamp_f32_f32_f32p_planar_ukernel& get_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla() {
    static kai_dwconv_clamp_f32_f32_f32p_planar_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    ukernel.get_dst_offset = kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    ukernel.get_dst_size = kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    ukernel.run_dwconv = kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    return ukernel;
}

const DepthwiseArray& get_depthwise_methods() {
    // FP32 kernels with 3x3 filter.
    static DepthwiseArray depthwise_methods{};
    depthwise_methods[0].name = "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla";
    depthwise_methods[0].rhs.get_rhs_packed_size = kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    depthwise_methods[0].rhs.pack = kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    depthwise_methods[0].is_supported = cpu_has_sme2;
    depthwise_methods[0].filter = {3, 3};

    const kai_dwconv_clamp_f32_f32_f32p_planar_ukernel& ukernel_f32 =
        get_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    depthwise_methods[0].data_type = DataType::FP32;
    depthwise_methods[0].acc_type = DataType::FP32;
    depthwise_methods[0].depthwise.get_m_step = ukernel_f32.get_m_step;
    depthwise_methods[0].depthwise.get_dst_size = ukernel_f32.get_dst_size;
    depthwise_methods[0].depthwise.get_dst_offset = ukernel_f32.get_dst_offset;
    depthwise_methods[0].depthwise.conv = ukernel_f32.run_dwconv;
    return depthwise_methods;
}

/// Test reference identification.
struct TestDataId {
    using DT = std::underlying_type_t<DataType>;
    MatMulShape in_shape;
    MatMulShape rhs_shape;
    Padding2D pad;
    DataType dt;
    DataType dt_acc;
    float clamp_rate;

    struct Hash {
        size_t operator()(const TestDataId& test_id) const {
            return                                                         //
                (MatMulShape::Hash{}(test_id.in_shape) << 0) ^             //
                (MatMulShape::Hash{}(test_id.rhs_shape) << 1) ^            //
                (Padding2D::Hash{}(test_id.pad) << 2) ^                    //
                (std::hash<DT>{}(static_cast<DT>(test_id.dt)) << 3) ^      //
                (std::hash<DT>{}(static_cast<DT>(test_id.dt_acc)) << 4) ^  //
                (std::hash<float>{}(test_id.clamp_rate) << 5);             //
        }
    };

private:
    friend bool operator==(const TestDataId& lhs, const TestDataId& rhs) {
        return                                 //
            lhs.in_shape == rhs.in_shape &&    //
            lhs.rhs_shape == rhs.rhs_shape &&  //
            lhs.pad == rhs.pad &&              //
            lhs.dt == rhs.dt &&                //
            lhs.dt_acc == rhs.dt_acc &&        //
            lhs.clamp_rate == rhs.clamp_rate;  //
    }
};

/// Test reference data
struct TestData {
    Buffer lhs;                ///< LHS input matrix
    Buffer rhs;                ///< RHS input matrix
    Buffer bias;               ///< Bias vector
    Buffer out;                ///< Reference depthwise result
    Buffer padding;            ///< Padding buffer
    Range<float> clamp_range;  ///< Clamp range
};

/// Generate reference data, caches it.
struct ReferenceGenerator {
    /// Retrieve reference data for the provided test identification
    static const TestData& get_test_reference(const TestDataId test_id, const MatMulShape& out_shape) {
        static std::unordered_map<TestDataId, TestData, TestDataId::Hash> m_data;
        if (const auto itr = m_data.find(test_id); itr != end(m_data)) {
            return itr->second;
        }

        return m_data[test_id] = generate_reference(test_id, out_shape);
    }

private:
    /// Return incremented seed value
    static size_t get_seed() {
        static size_t seed = 0;
        return seed++;
    }

    /// Generate reference data.
    // NOTE : This block is currently FP32 specific - it is not datatype generic
    static TestData generate_reference(const TestDataId& test_id, const MatMulShape& out_shape) {
        const auto& [in_shape, rhs_shape, pad, dt, acc_dt, clamp_rate] = test_id;

        // Generate random input data
        Buffer lhs = fill_matrix_random(in_shape.m, in_shape.n * in_shape.k, DataFormat(dt), get_seed());
        Buffer rhs = fill_matrix_random(rhs_shape.m, rhs_shape.n * rhs_shape.k, DataFormat(dt), get_seed());
        Buffer bias = fill_matrix_random(1, out_shape.k, DataFormat(dt), get_seed());

        // Call reference function
        Buffer out = depthwise_reference<float>(
            1, in_shape.m, in_shape.n, in_shape.k, rhs_shape.m, rhs_shape.n, lhs.data(), rhs.data(), bias.data(), pad);

        const auto [min, max] =
            find_clamp_range(dt, out.data(), out_shape.m * out_shape.n * out_shape.k, 1.0F - clamp_rate);
        Buffer out_clamped = clamp(dt, out.data(), out_shape.m * out_shape.n * out_shape.k, min, max);

        // Populate reference data
        TestData test_reference;
        test_reference.lhs = std::move(lhs);
        test_reference.rhs = std::move(rhs);
        test_reference.bias = std::move(bias);
        test_reference.out = std::move(out_clamped);
        test_reference.clamp_range = {min, max};
        return test_reference;
    };
};

/// Perform RHS packing for depthwise
Buffer pack_rhs(const RhsPackDepthwiseKernel& kernel, const MatMulShape& shape, const TestData& reference) {
    // Calculate size, and allocate buffer
    const size_t dst_size = kernel.get_rhs_packed_size(shape.m, shape.n, shape.k);
    Buffer dst(dst_size);

    // RHS Pack API is subject to change.
    abi_check(
        kernel.pack, shape.m, shape.n, shape.m, shape.n, shape.k, reference.rhs.data(), reference.bias.data(),
        dst.data());
    return dst;
}

/// Perform Depthwise Operation using main kernel.
Buffer dwconv(
    const DepthwisePlanarKernel& kernel, const Rect& portion, const MatMulShape& in_shape, const MatMulShape& out_shape,
    const Padding2D pad, const TestData& reference, const Buffer& rhs_packed, Range<float> clamp_range, DataType type) {
    const size_t dst_size = kernel.get_dst_size(out_shape.m, out_shape.n, out_shape.k);
    Buffer dst(dst_size);

    const size_t dt_size_bytes = data_type_size_in_bits(type) / 8;
    const size_t stride_in_row = in_shape.n * in_shape.k * dt_size_bytes;
    const size_t dst_stride_row = out_shape.n * out_shape.k * dt_size_bytes;
    const size_t stride_col = out_shape.k * dt_size_bytes;

    // Loop the following. M-Step rows are handled at a time.
    for (size_t out_row = portion.start_row(); out_row < portion.end_row(); out_row += kernel.get_m_step()) {
        const int start_in_row = out_row - pad.top;
        const size_t pad_top = (start_in_row < 0) ? (-start_in_row) : 0;
        const size_t in_row = (start_in_row < 0) ? 0 : start_in_row;

        const size_t valid_input_rows = (in_row < in_shape.m) ? (in_shape.m - in_row) : 0;
        const size_t valid_out_rows = (out_shape.m - out_row);

        abi_check(
            kernel.conv, reference.lhs.data() + (in_row * stride_in_row), rhs_packed.data(),
            dst.data() + (out_row * dst_stride_row), stride_in_row, stride_col, dst_stride_row, stride_col,
            valid_input_rows, valid_out_rows, pad.left, pad_top, 0.f, clamp_range.min, clamp_range.max);
    }

    return dst;
}
}  // namespace

/// End-to-end test for depthwise kernels
TEST_P(DepthwisePlanarTest, Output) {
    const auto& [method, in_shape, padding, clamp_rate] = GetParam();
    if (not method.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    // Calculate Shapes.
    const int out_height = (in_shape.m + padding.top + padding.bottom + 1 - method.filter.first);
    const int out_width = (in_shape.n + padding.left + padding.right + 1 - method.filter.second);
    ASSERT_TRUE(out_height > 0 && out_width > 0);

    const size_t dt_size_bytes = data_type_size_in_bits(method.data_type) / 8;
    MatMulShape rhs_shape = {method.filter.first, method.filter.second, in_shape.k};
    MatMulShape out_shape = {static_cast<size_t>(out_height), static_cast<size_t>(out_width), (in_shape.k)};

    // 1. Calculate reference.
    const TestData& test_data = ReferenceGenerator::get_test_reference(
        {in_shape, rhs_shape, padding, method.data_type, method.acc_type, clamp_rate}, out_shape);

    // 2. Pack RHS (Weights+Bias)
    Buffer rhs_packed = pack_rhs(method.rhs, rhs_shape, test_data);
    const MatrixPortion out_portion{0, 0, 1, 1};
    const Rect portion = out_portion.compute_portion(
        out_shape.m, out_shape.n * out_shape.k, method.depthwise.get_m_step(), (rhs_packed.size() / dt_size_bytes));

    // 3. Run Depthwise Kernel.
    Buffer out = dwconv(
        method.depthwise, portion, in_shape, out_shape, padding, test_data, rhs_packed, test_data.clamp_range,
        method.data_type);

    // 4. Compare with reference result.
    DefaultMismatchHandler handler(0, 0.0001, 0, 0.001);
    const auto success = compare(
        out.data(), test_data.out.data(), DataType::FP32, out_shape.m, out_shape.n * out_shape.k, portion, handler);
    ASSERT_TRUE(success);
}

/// Name generator for test case
[[maybe_unused]] static void PrintTo(const DepthwiseParamsParams& param, std::ostream* os) {
    const auto& [method, shape, padding, clamp_rate] = param;
    *os << method.name << "__";
    PrintTo(shape, os);
    *os << "__";
    PrintTo(padding, os);
    *os << "__";
    *os << "__clamp_rate_" << static_cast<int>(clamp_rate * 100);
}

///  Test parameter listing
INSTANTIATE_TEST_SUITE_P(
    Depthwise, DepthwisePlanarTest,
    testing::Combine(
        testing::ValuesIn(get_depthwise_methods()),  //
        testing::ValuesIn({
            // clang-format off
            // IN_HEIGHT, IN_WIDTH, IN_CHANNELS
            MatMulShape{  4,    4,   1},   //
            MatMulShape{  8,    4,   16},  //
            MatMulShape{  96,   33,  37},  //
            MatMulShape{  99,   22,  51},  //
            MatMulShape{  127,  127, 127}, //
            // clang-format on
        }),
        testing::ValuesIn({
            // clang-format off
            // pad_left, pad_right, pad_top, pad_bottom
            Padding2D{0, 0, 0, 0},
            Padding2D{0, 1, 0, 1},
            Padding2D{1, 1, 1, 1},
            Padding2D{5, 11, 7, 3},
            // clang-format on
        }),
        testing::ValuesIn(std::initializer_list<float>{0.0F, 0.1F, 0.5F})),  //
    testing::PrintToStringParamName());

}  // namespace kai::test
