/*
 * Copyright (c) 2017-2024 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ACL_TESTS_VALIDATION_FIXTURES_GEMMLOWPFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_GEMMLOWPFIXTURE_H

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/utils/quantization/AsymmHelpers.h"
#include "tests/validation/Helpers.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/GEMMLowp.h"

#include <cstdint>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{

template <typename U>
void fill(U &&tensor, int i)
{
    ARM_COMPUTE_ASSERT(is_data_type_quantized(tensor.data_type()));
    library->fill_tensor_uniform(tensor, i);
}

template <typename U>
void fill_bias_s32(U &&tensor, int i, int32_t min, int32_t max)
{
    ARM_COMPUTE_ASSERT(tensor.data_type() == DataType::S32);
    std::uniform_int_distribution<int32_t> distribution(min, max);
    library->fill(tensor, distribution, i);
}

/** Information about how to fill tensors */
struct TensorFillInfo
{
    // Bias fill range. Default values are arbitrary
    int32_t min_bias {-20000};
    int32_t max_bias {20000};
    // Optional extra hash to randomize tensor filling
    int32_t hash     {0};
};

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d, bool reinterpret_output_as_3d, typename OutputType, bool is_fused = false, bool run_twice = false>
TensorType compute_gemmlowp_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo,
                                   const QuantizationInfo& output_qinfo, DataType data_type_a = DataType::QASYMM8, DataType data_type_b = DataType::QASYMM8,
                                   GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo(), bool reshape_b_only_on_first_run = false, const TensorFillInfo& finfo = TensorFillInfo() )
{
    ARM_COMPUTE_ASSERT(is_data_type_quantized_asymmetric(data_type_a));
    ARM_COMPUTE_ASSERT(data_type_a == data_type_b);
    // Create tensors
    const DataType data_type_output = output_stage.type == GEMMLowpOutputStageType::NONE ? DataType::S32 : data_type_a;

    TensorType a      = create_tensor<TensorType>(shape_a, data_type_a, 1, a_qinfo);
    TensorType b      = create_tensor<TensorType>(shape_b, data_type_b, 1, b_qinfo); // gemm output before output stage mismatch if i pass data_layout_output here. to be investigated
    TensorType output = create_tensor<TensorType>(shape_output, data_type_output, 1, output_qinfo /* output_qinfo will be ignored when output stage type is None */);

    TensorType bias;
    if(is_fused)
    {
        TensorShape bias_shape(shape_b[0]);
        bias = create_tensor<TensorType>(bias_shape, DataType::S32, 1);
    }

    // Create and configure function
    // The GEMMinfo includes the values of the depth in case of reinterpreted 3d input/output
    FunctionType gemmlowp;
    gemmlowp.configure(&a, &b, is_fused ? &bias : nullptr, &output, GEMMInfo(false, false, reshape_b_only_on_first_run, (reinterpret_output_as_3d ? shape_output[2] : 0), reinterpret_input_as_3d, false,
                                                                             output_stage));

    ARM_COMPUTE_ASSERT(a.info()->is_resizable());
    ARM_COMPUTE_ASSERT(b.info()->is_resizable());
    ARM_COMPUTE_ASSERT(output.info()->is_resizable());

    add_padding_x({ &a, &b, &output });

    // Allocate tensors
    a.allocator()->allocate();
    b.allocator()->allocate();
    output.allocator()->allocate();

    ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
    ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
    ARM_COMPUTE_ASSERT(!output.info()->is_resizable());

    // Fill tensors
    fill(AccessorType(a), 0 + finfo.hash);
    fill(AccessorType(b), 1 + finfo.hash);

    if(is_fused)
    {
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        bias.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        fill_bias_s32(AccessorType(bias), 2 + finfo.hash, finfo.min_bias, finfo.max_bias);
    }

    // Run with variable inputs.
    if(run_twice)
    {
        gemmlowp.run();
        fill(AccessorType(a), 3 + finfo.hash); // Fill tensors with new seed after run
        fill(AccessorType(b), 4 + finfo.hash);
        if(is_fused)
        {
            fill_bias_s32(AccessorType(bias), 5 + finfo.hash, finfo.min_bias, finfo.max_bias);
        }
    }

    // Compute GEMM function
    gemmlowp.run();
    return output;
}

template <bool reinterpret_input_as_3d, typename TI = uint8_t, typename TW = uint8_t, bool pretranspose_A = false, bool pretranspose_B = false, bool run_twice = false>
SimpleTensor<int32_t> compute_gemmlowp_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo,
                                                 DataType data_type_a = DataType::QASYMM8, DataType data_type_b = DataType::QASYMM8, const TensorFillInfo& finfo = TensorFillInfo())
{
    ARM_COMPUTE_ASSERT(is_data_type_quantized_asymmetric(data_type_a));
    ARM_COMPUTE_ASSERT(data_type_a == data_type_b);
    TensorShape shape_a_to_use = shape_a;
    if(reinterpret_input_as_3d)
    {
        // Collapse the second and third dimension if the input is 3D
        shape_a_to_use.collapse(2U, 1U);
    }

    // Create reference
    SimpleTensor<TI> a{ shape_a_to_use, data_type_a, 1, a_qinfo };
    SimpleTensor<TW> b{ shape_b, data_type_b, 1, b_qinfo };

    TensorShape shape_a_to_use_transposed{ shape_a_to_use };
    TensorShape shape_b_transposed{ shape_b };

    shape_a_to_use_transposed.set(0, shape_a_to_use[1]);
    shape_a_to_use_transposed.set(1, shape_a_to_use[0]);
    shape_b_transposed.set(0, shape_b[1]);
    shape_b_transposed.set(1, shape_b[0]);

    SimpleTensor<TI> a_transposed{ shape_a_to_use_transposed, data_type_a, 1, a_qinfo };
    SimpleTensor<TW> b_transposed{ shape_b_transposed, data_type_b, 1, b_qinfo };

    // Fill reference
    fill(a, 0 + finfo.hash);
    fill(b, 1 + finfo.hash);

    // Transpose reference if required
    /* Note: Assuming the usual batch matmul dimensions A = (B x M x K), B = (B x K x N), if pretranspose_A is set to true, then A is assumed to be (B x K x M),
       therefore, A must be pre-transposed before passing it to the fixture. And, we transpose A again in the fixture to make it (B x M x K)
       in order to be able to call reference implementation that works with (B x M x K) input.
       Similarly, if pretranspose_B is set to true, then B is assumed to be (B x N x K), B must be pre-transposed before passing it to the fixture. */
    if(pretranspose_A)
    {
        transpose_matrix<TI>(a, a_transposed);
    }

    if(pretranspose_B)
    {
        transpose_matrix<TW>(b, b_transposed);
    }

    // Run with variable inputs.
    const int32_t a_offset = a_qinfo.uniform().offset;
    const int32_t b_offset = b_qinfo.uniform().offset;
    if(run_twice)
    {
        reference::gemmlowp_matrix_multiply_core<int32_t, TI, TW>((pretranspose_A ? a_transposed : a), (pretranspose_B ? b_transposed : b), shape_output, a_offset, b_offset);
        fill((pretranspose_A) ? a_transposed : a, 3 + finfo.hash);
        fill((pretranspose_B) ? b_transposed : b, 4 + finfo.hash);
    }

    return reference::gemmlowp_matrix_multiply_core<int32_t, TI, TW>((pretranspose_A ? a_transposed : a), (pretranspose_B ? b_transposed : b), shape_output, a_offset, b_offset);
}
} // namespace

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, bool run_twice = false>
class GEMMLowpMatrixMultiplyCoreValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset)
    {
        const auto a_qinfo = QuantizationInfo(1.0f / 255, a_offset);
        const auto b_qinfo = QuantizationInfo(1.0f / 255, b_offset);
        _target    = compute_target(shape_a, shape_b, shape_output, a_qinfo, b_qinfo);
        _reference = compute_reference(shape_a, shape_b, shape_output, a_qinfo, b_qinfo);
    }

protected:
    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo)
    {
        const auto output_qinfo = QuantizationInfo(); // No output stage
        return compute_gemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, int32_t, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, output_qinfo);
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo)
    {
        return compute_gemmlowp_reference<reinterpret_input_as_3d, uint8_t, uint8_t, false, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo);
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, typename TI = uint8_t, typename TW = uint8_t, bool run_twice = false>
class GEMMLowpMatrixMultiplyCoreFusedOffsetOutputGenericValidationFixture : public framework::Fixture
{
public:
    /** Dynamically initialize the quantization info with saturation awareness
     */
    template <typename T>
    static void setup_quantization(DataType data_type, const TensorShape& shape_a, const TensorShape& shape_b, QuantizationInfo& a_qinfo, QuantizationInfo& b_qinfo, QuantizationInfo& output_qinfo, TensorFillInfo& finfo)
    {
        // This hash is used by random generators. There may be hash collisions but
        // this is intentional as it's a very easy way to make the the current
        // random generation process almost different for many test configurations,
        // which were using the same set of values before.
        finfo.hash = shape_a[0] + shape_a[1] + shape_b[0] + shape_b[1];

        const int32_t t_max = static_cast<int32_t>(std::numeric_limits<T>::max());
        const int32_t t_min = static_cast<int32_t>(std::numeric_limits<T>::min());

        std::mt19937                           generator(library->seed() + finfo.hash);
        std::uniform_real_distribution<float>  distribution_float(-5.0f, 3.0f);
        std::uniform_int_distribution<int32_t> distribution_t(t_min, t_max);

        const float scale_lhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]
        const float scale_rhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]

        const int32_t offset_lhs = distribution_t(generator);
        const int32_t offset_rhs = distribution_t(generator);

        a_qinfo = QuantizationInfo(scale_lhs, offset_lhs);
        b_qinfo = QuantizationInfo(scale_rhs, offset_rhs);

        // reinterpret_input_as_3d or reinterpret_output_as_3d can be ignored, as the underlying gemm / matmul computation
        // is equivalent to a standard 2D one with m-n-k dimensions
        const int m = shape_a.y();
        const int n = shape_b.x();
        const int k = shape_a.x();

        const float bias_fraction = 0.5f; // We enabled is_fused in compute_gemmlowp_target below, thus bias is included

        QuantizationHint q_hint = suggest_matmul_dst_q_info_and_bias(a_qinfo, b_qinfo, m, n, k, data_type, bias_fraction);
        output_qinfo            = q_hint.q_info;
        finfo.min_bias          = q_hint.bias_min;
        finfo.max_bias          = q_hint.bias_max;

        // Both target and reference implementations use negated offsets, i.e.
        //      float_val = (int_val + offset) * scale
        // instead of
        //      float_val = (int_val - offset) * scale
        // as usual. Therefore, after calculating the output quantization above, we
        // negate the offsets of inputs' offsets.
        a_qinfo = QuantizationInfo(scale_lhs, -offset_lhs);
        b_qinfo = QuantizationInfo(scale_rhs, -offset_rhs);
    }

    /** Initialize output stage info from quantization info */
    static Status init_gemmlowp_output_stage_info(
                                        DataType                data_type,
                                        const QuantizationInfo& a_qinfo,
                                        const QuantizationInfo& b_qinfo,
                                        const QuantizationInfo& output_qinfo,
                                        GEMMLowpOutputStageType type,
                                        GEMMLowpOutputStageInfo &gemmlowp_output_stage_info)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(!is_data_type_quantized_asymmetric(data_type));

        const UniformQuantizationInfo aq_unif   = a_qinfo.uniform();
        const UniformQuantizationInfo bq_unif   = b_qinfo.uniform();
        const UniformQuantizationInfo oq_unif   = output_qinfo.uniform();

        float   multiplier = (aq_unif.scale * bq_unif.scale) / oq_unif.scale;
        int32_t int_multiplier;
        int32_t shift;

        ARM_COMPUTE_RETURN_ON_ERROR(
            quantization::calculate_quantized_multiplier(multiplier, &int_multiplier, &shift));

        int32_t type_min             = 0;
        int32_t type_max             = 0;
        std::tie(type_min, type_max) = quantization::get_quantized_asymmetric_output_min_max(output_qinfo, ActivationLayerInfo(), data_type);

        gemmlowp_output_stage_info.gemmlowp_real_multiplier = multiplier;
        gemmlowp_output_stage_info.gemmlowp_multiplier = int_multiplier;
        gemmlowp_output_stage_info.gemmlowp_multipliers = { int_multiplier };
        gemmlowp_output_stage_info.gemmlowp_shift      = shift;
        gemmlowp_output_stage_info.gemmlowp_shifts     = { shift };
        gemmlowp_output_stage_info.gemmlowp_offset     = oq_unif.offset;
        gemmlowp_output_stage_info.type                = type;
        gemmlowp_output_stage_info.gemmlowp_min_bound  = type_min;
        gemmlowp_output_stage_info.gemmlowp_max_bound  = type_max;

        return Status{};
    }

    /** Currently this fixture only tests the following data type configurations:
     *
     * 1. a and b are of the same data type
     * 2. The data type is quantized asymmetric
     *
     */
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, GEMMLowpOutputStageType output_stage_type, DataType data_type,
               bool reshape_b_only_on_first_run)
    {
        ARM_COMPUTE_ASSERT(output_stage_type != GEMMLowpOutputStageType::NONE);
        ARM_COMPUTE_ASSERT(is_data_type_quantized_asymmetric(data_type));

        // Randomized dynamic quantization: randomize quantization info in a way that ensures no result saturation
        // most of the time
        QuantizationInfo a_qinfo;
        QuantizationInfo b_qinfo;
        QuantizationInfo output_qinfo;
        TensorFillInfo finfo;
        setup_quantization<TI>(data_type, shape_a, shape_b, a_qinfo, b_qinfo, output_qinfo, finfo);

        GEMMLowpOutputStageInfo output_stage;
        init_gemmlowp_output_stage_info(data_type, a_qinfo, b_qinfo, output_qinfo, output_stage_type, output_stage);

        _reference = compute_reference(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, data_type, data_type, output_stage, finfo);
        _target    = compute_target(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, output_qinfo, data_type, data_type, output_stage, reshape_b_only_on_first_run, finfo);
    }

protected:
    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo, const QuantizationInfo& output_qinfo,
                              DataType data_type_a, DataType data_type_b, const GEMMLowpOutputStageInfo& output_stage, bool reshape_b_only_on_first_run = false, const TensorFillInfo& finfo = TensorFillInfo())
    {
        return compute_gemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, qasymm8_t, true, run_twice>(shape_a, shape_b, shape_output, a_qinfo,
                b_qinfo, output_qinfo, data_type_a, data_type_b, output_stage, reshape_b_only_on_first_run, finfo);
    }

    SimpleTensor<TI> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo,
                                       DataType data_type_a, DataType data_type_b, const GEMMLowpOutputStageInfo& output_stage, const TensorFillInfo& finfo = TensorFillInfo())
    {
        SimpleTensor<int32_t> output = compute_gemmlowp_reference<reinterpret_input_as_3d, TI, TW, false, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, data_type_a, data_type_b, finfo);

        TensorShape           bias_shape(shape_b[0]);
        SimpleTensor<int32_t> bias{ bias_shape, DataType::S32, 1 };
        (run_twice) ? fill_bias_s32(bias, 5 + finfo.hash, finfo.min_bias, finfo.max_bias) : fill_bias_s32(bias, 2 + finfo.hash, finfo.min_bias, finfo.max_bias); // Fill bias with same seed as last run of gemmlowp_target

        switch(output_stage.type)
        {
            case GEMMLowpOutputStageType::QUANTIZE_DOWN:
                return reference::gemmlowp_quantize_down_scale<int32_t, TW>(output, bias,
                                                                            output_stage.gemmlowp_offset, output_stage.gemmlowp_multipliers, output_stage.gemmlowp_shifts, output_stage.gemmlowp_min_bound, output_stage.gemmlowp_max_bound);
                break;
            case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
                return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, TW>(output, bias,
                                                                                          output_stage.gemmlowp_multipliers, output_stage.gemmlowp_shifts, output_stage.gemmlowp_offset, output_stage.gemmlowp_min_bound, output_stage.gemmlowp_max_bound);
                break;
            default:
                ARM_COMPUTE_ERROR("Not Supported!");
        }
    }

    TensorType       _target{};
    SimpleTensor<TI> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, typename TI = uint8_t, typename TW = uint8_t>
class GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture : public
    GEMMLowpMatrixMultiplyCoreFusedOffsetOutputGenericValidationFixture<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, TI, TW>
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, GEMMLowpOutputStageType output_stage_type, DataType data_type)
    {
        GEMMLowpMatrixMultiplyCoreFusedOffsetOutputGenericValidationFixture<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, TI, TW>::setup(shape_a, shape_b,
                shape_output, output_stage_type, data_type, false /* reshape_b_only_on_first_run */);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpQuantizeDownInt32ToUint8ScaleValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        _target    = compute_target(shape, result_offset, result_mult_int, result_shift, min, max, add_bias);
        _reference = compute_reference(shape, result_offset, result_mult_int, result_shift, min, max, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_int_distribution<> distribution(-6000, 6000);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        TensorShape shape_bias(shape[0]);

        // Create tensors
        TensorType a = create_tensor<TensorType>(shape, DataType::S32, 1);
        TensorType b = create_tensor<TensorType>(shape_bias, DataType::S32, 1);
        TensorType c = create_tensor<TensorType>(shape, DataType::QASYMM8, 1);

        // Create and configure function
        FunctionType            output_stage;
        GEMMLowpOutputStageInfo output_stage_info = GEMMLowpOutputStageInfo();
        output_stage_info.type                    = GEMMLowpOutputStageType::QUANTIZE_DOWN;
        output_stage_info.gemmlowp_offset         = result_offset;
        output_stage_info.gemmlowp_multiplier     = result_mult_int;
        output_stage_info.gemmlowp_shift          = result_shift;
        output_stage_info.gemmlowp_min_bound      = min;
        output_stage_info.gemmlowp_max_bound      = max;
        output_stage_info.output_data_type        = DataType::QASYMM8;
        output_stage.configure(&a, add_bias ? &b : nullptr, &c, output_stage_info);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());

        // Fill tensor
        fill(AccessorType(a), 0);

        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(b.info()->is_resizable());

            // Allocate bias tensor
            b.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!b.info()->is_resizable());

            // Fill tensor
            fill(AccessorType(b), 1);
        }

        // Compute GEMM function
        output_stage.run();
        return c;
    }

    SimpleTensor<uint8_t> compute_reference(const TensorShape &shape, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        // Create reference
        TensorShape shape_bias(shape[0]);

        SimpleTensor<int32_t> a{ shape, DataType::S32, 1 };
        SimpleTensor<int32_t> b{ shape_bias, DataType::S32, 1 };

        // Fill reference
        fill(a, 0);

        const std::vector<int32_t> result_mult_int_vec = { result_mult_int };
        const std::vector<int32_t> result_shift_vec    = { result_shift };

        if(add_bias)
        {
            // Fill bias
            fill(b, 1);

            return reference::gemmlowp_quantize_down_scale<int32_t, uint8_t>(a, b, result_offset, result_mult_int_vec, result_shift_vec, min, max);
        }
        else
        {
            return reference::gemmlowp_quantize_down_scale<int32_t, uint8_t>(a, result_offset, result_mult_int_vec, result_shift_vec, min, max);
        }
    }

    TensorType            _target{};
    SimpleTensor<uint8_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpQuantizeDownInt32ToInt8ScaleValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        _target    = compute_target(shape, result_offset, result_mult_int, result_shift, min, max, add_bias);
        _reference = compute_reference(shape, result_offset, result_mult_int, result_shift, min, max, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_int_distribution<> distribution(-6000, 6000);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        TensorShape shape_bias(shape[0]);

        // Create tensors
        TensorType a = create_tensor<TensorType>(shape, DataType::S32, 1);
        TensorType b = create_tensor<TensorType>(shape_bias, DataType::S32, 1);
        TensorType c = create_tensor<TensorType>(shape, DataType::QASYMM8_SIGNED, 1);

        // Create and configure function
        FunctionType            output_stage;
        GEMMLowpOutputStageInfo output_stage_info = GEMMLowpOutputStageInfo();
        output_stage_info.type                    = GEMMLowpOutputStageType::QUANTIZE_DOWN;
        output_stage_info.gemmlowp_offset         = result_offset;
        output_stage_info.gemmlowp_multiplier     = result_mult_int;
        output_stage_info.gemmlowp_shift          = result_shift;
        output_stage_info.gemmlowp_min_bound      = min;
        output_stage_info.gemmlowp_max_bound      = max;
        output_stage_info.output_data_type        = DataType::QASYMM8_SIGNED;
        output_stage.configure(&a, add_bias ? &b : nullptr, &c, output_stage_info);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());

        // Fill tensor
        fill(AccessorType(a), 0);

        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(b.info()->is_resizable());

            // Allocate bias tensor
            b.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!b.info()->is_resizable());

            // Fill tensor
            fill(AccessorType(b), 1);
        }

        // Compute GEMM function
        output_stage.run();
        return c;
    }

    SimpleTensor<int8_t> compute_reference(const TensorShape &shape, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        // Create reference
        TensorShape shape_bias(shape[0]);

        SimpleTensor<int32_t> a{ shape, DataType::S32, 1 };
        SimpleTensor<int32_t> b{ shape_bias, DataType::S32, 1 };

        // Fill reference
        fill(a, 0);

        const std::vector<int32_t> result_mult_int_vec = { result_mult_int };
        const std::vector<int32_t> result_shift_vec    = { result_shift };

        if(add_bias)
        {
            // Fill bias
            fill(b, 1);

            return reference::gemmlowp_quantize_down_scale<int32_t, int8_t>(a, b, result_offset, result_mult_int_vec, result_shift_vec, min, max);
        }
        else
        {
            return reference::gemmlowp_quantize_down_scale<int32_t, int8_t>(a, result_offset, result_mult_int_vec, result_shift_vec, min, max);
        }
    }

    TensorType           _target{};
    SimpleTensor<int8_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max, bool add_bias)
    {
        _target    = compute_target(shape, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max, add_bias);
        _reference = compute_reference(shape, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_int_distribution<> distribution(-6000, 6000);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max, bool add_bias)
    {
        TensorShape shape_bias(shape[0]);

        // Create tensors
        TensorType a = create_tensor<TensorType>(shape, DataType::S32, 1);
        TensorType b = create_tensor<TensorType>(shape_bias, DataType::S32, 1);
        TensorType c = create_tensor<TensorType>(shape, DataType::QASYMM8_SIGNED, 1);

        // Create and configure function
        FunctionType output_stage;
        output_stage.configure(&a, add_bias ? &b : nullptr, &c, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());

        // Fill tensor
        fill(AccessorType(a), 0);

        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(b.info()->is_resizable());

            // Allocate bias tensor
            b.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!b.info()->is_resizable());

            // Fill tensor
            fill(AccessorType(b), 1);
        }

        // Compute GEMM function
        output_stage.run();
        return c;
    }

    SimpleTensor<int8_t> compute_reference(const TensorShape &shape, int32_t result_fixed_point_multiplier, int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max,
                                           bool add_bias)
    {
        // Create reference
        TensorShape shape_bias(shape[0]);

        SimpleTensor<int32_t> a{ shape, DataType::S32, 1 };
        SimpleTensor<int32_t> b{ shape_bias, DataType::S32, 1 };

        // Fill reference
        fill(a, 0);

        const std::vector<int32_t> result_fixed_point_multiplier_vec = { result_fixed_point_multiplier };
        const std::vector<int32_t> result_shift_vec                  = { result_shift };

        if(add_bias)
        {
            // Fill bias
            fill(b, 1);

            return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, int8_t>(a, b, result_fixed_point_multiplier_vec, result_shift_vec, result_offset_after_shift, min, max);
        }
        else
        {
            return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, int8_t>(a, result_fixed_point_multiplier_vec, result_shift_vec, result_offset_after_shift, min, max);
        }
    }

    TensorType           _target{};
    SimpleTensor<int8_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max, bool add_bias)
    {
        _target    = compute_target(shape, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max, add_bias);
        _reference = compute_reference(shape, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_int_distribution<> distribution(-6000, 6000);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max, bool add_bias)
    {
        TensorShape shape_bias(shape[0]);

        // Create tensors
        TensorType a = create_tensor<TensorType>(shape, DataType::S32, 1);
        TensorType b = create_tensor<TensorType>(shape_bias, DataType::S32, 1);
        TensorType c = create_tensor<TensorType>(shape, DataType::QASYMM8, 1);

        // Create and configure function
        FunctionType output_stage;
        output_stage.configure(&a, add_bias ? &b : nullptr, &c, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());

        // Fill tensor
        fill(AccessorType(a), 0);

        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(b.info()->is_resizable());

            // Allocate bias tensor
            b.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!b.info()->is_resizable());

            // Fill tensor
            fill(AccessorType(b), 1);
        }

        // Compute GEMM function
        output_stage.run();
        return c;
    }

    SimpleTensor<uint8_t> compute_reference(const TensorShape &shape, int32_t result_fixed_point_multiplier, int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max,
                                            bool add_bias)
    {
        // Create reference
        TensorShape shape_bias(shape[0]);

        SimpleTensor<int32_t> a{ shape, DataType::S32, 1 };
        SimpleTensor<int32_t> b{ shape_bias, DataType::S32, 1 };

        // Fill reference
        fill(a, 0);

        const std::vector<int32_t> result_fixed_point_multiplier_vec = { result_fixed_point_multiplier };
        const std::vector<int32_t> result_shift_vec                  = { result_shift };

        if(add_bias)
        {
            // Fill bias
            fill(b, 1);

            return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, uint8_t>(a, b, result_fixed_point_multiplier_vec, result_shift_vec, result_offset_after_shift, min, max);
        }
        else
        {
            return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, uint8_t>(a, result_fixed_point_multiplier_vec, result_shift_vec, result_offset_after_shift, min, max);
        }
    }

    TensorType            _target{};
    SimpleTensor<uint8_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GEMMLowpQuantizeDownInt32ScaleByFloatValidationFixture : public framework::Fixture
{
public:
    void setup(DataType data_type, TensorShape shape, float result_real_multiplier, int32_t result_offset, int32_t min, int32_t max, bool add_bias)
    {
        _target    = compute_target(data_type, shape, result_real_multiplier, result_offset, min, max, add_bias);
        _reference = compute_reference(shape, result_real_multiplier, result_offset, min, max, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        // To avoid data all being clampped
        std::uniform_int_distribution<> distribution(-500, 500);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(DataType data_type, const TensorShape &shape, float result_multiplier, int32_t result_offset, int32_t min, int32_t max, bool add_bias)
    {
        TensorShape shape_bias(shape[0]);

        // Create tensors
        TensorType a = create_tensor<TensorType>(shape, DataType::S32, 1);
        TensorType b = create_tensor<TensorType>(shape_bias, DataType::S32, 1);
        TensorType c = create_tensor<TensorType>(shape, data_type, 1);

        // create output stage info
        GEMMLowpOutputStageInfo info;
        info.gemmlowp_max_bound       = max;
        info.gemmlowp_min_bound       = min;
        info.gemmlowp_real_multiplier = result_multiplier;
        info.gemmlowp_offset          = result_offset;
        info.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT;
        info.output_data_type         = data_type;

        // Create and configure function
        FunctionType output_stage;
        output_stage.configure(&a, add_bias ? &b : nullptr, &c, info);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());

        // Fill tensor
        fill(AccessorType(a), 0);

        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(b.info()->is_resizable());

            // Allocate bias tensor
            b.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!b.info()->is_resizable());

            // Fill tensor
            fill(AccessorType(b), 1);
        }

        // Compute GEMM function
        output_stage.run();
        return c;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, float_t result_real_multiplier, int32_t result_offset, int32_t min, int32_t max, bool add_bias)
    {
        // Create reference
        TensorShape shape_bias(shape[0]);

        SimpleTensor<int32_t> a{ shape, DataType::S32, 1 };
        SimpleTensor<int32_t> b{ shape_bias, DataType::S32, 1 };

        // Fill reference
        fill(a, 0);

        const std::vector<float_t> result_float_multiplier_vec = { result_real_multiplier };

        if(add_bias)
        {
            // Fill bias
            fill(b, 1);

            return reference::gemmlowp_quantize_down_scale_by_float<int32_t, T>(a, b, result_float_multiplier_vec, result_offset, min, max);
        }
        else
        {
            return reference::gemmlowp_quantize_down_scale_by_float<int32_t, T>(a, result_float_multiplier_vec, result_offset, min, max);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        _target    = compute_target(shape, result_fixedpoint_multiplier, result_shift, min, max, add_bias);
        _reference = compute_reference(shape, result_fixedpoint_multiplier, result_shift, min, max, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_int_distribution<> distribution(-6000, 6000);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t min, int32_t max, bool add_bias)
    {
        TensorShape shape_bias(shape[0]);

        // Create tensors
        TensorType a = create_tensor<TensorType>(shape, DataType::S32, 1);
        TensorType b = create_tensor<TensorType>(shape_bias, DataType::S32, 1);
        TensorType c = create_tensor<TensorType>(shape, DataType::QSYMM16, 1);

        // Create and configure function
        FunctionType output_stage;
        output_stage.configure(&a, add_bias ? &b : nullptr, &c, result_fixedpoint_multiplier, result_shift, min, max);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());

        // Fill tensor
        fill(AccessorType(a), 0);

        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(b.info()->is_resizable());

            // Allocate bias tensor
            b.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!b.info()->is_resizable());

            // Fill tensor
            fill(AccessorType(b), 1);
        }

        // Compute GEMM function
        output_stage.run();
        return c;
    }

    SimpleTensor<int16_t> compute_reference(const TensorShape &shape, int32_t result_fixed_point_multiplier, int32_t result_shift, int32_t min, int32_t max,
                                            bool add_bias)
    {
        // Create reference
        TensorShape shape_bias(shape[0]);

        SimpleTensor<int32_t> a{ shape, DataType::S32, 1 };
        SimpleTensor<int32_t> b{ shape_bias, DataType::S32, 1 };

        // Fill reference
        fill(a, 0);

        const std::vector<int32_t> result_fixed_point_multiplier_vec = { result_fixed_point_multiplier };
        const std::vector<int32_t> result_shift_vec                  = { result_shift };

        if(add_bias)
        {
            // Fill bias
            fill(b, 1);

            return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, int16_t>(a, b, result_fixed_point_multiplier_vec, result_shift_vec, 0, min, max);
        }
        else
        {
            return reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, int16_t>(a, result_fixed_point_multiplier_vec, result_shift_vec, 0, min, max);
        }
    }

    TensorType            _target{};
    SimpleTensor<int16_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyReshapedValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0, bool interleave_lhs,
               bool interleave_rhs, DataType data_type)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = false;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = true;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
                std::uniform_int_distribution<> distribution(1, 254);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::QASYMM8_SIGNED:
            {
                std::uniform_int_distribution<> distribution(-127, 126);
                library->fill(tensor, distribution, i);
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, DataType data_type)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs_reshaped.info(), rhs_reshaped.info(), dst.info(), lhs_info, rhs_info, GEMMReshapeInfo(M, N, K));

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped }, { ACL_SRC_1, &rhs_reshaped }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        switch(data_type)
        {
            case DataType::QASYMM8:
            {
                // Create reference
                SimpleTensor<uint8_t> lhs{ lhs_shape, data_type, 1 };
                SimpleTensor<uint8_t> rhs{ rhs_shape, data_type, 1 };

                // Fill reference
                fill(lhs, 0);
                fill(rhs, 1);

                return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
            }
            case DataType::QASYMM8_SIGNED:
            {
                // Create reference
                SimpleTensor<int8_t> lhs{ lhs_shape, data_type, 1 };
                SimpleTensor<int8_t> rhs{ rhs_shape, data_type, 1 };

                // Fill reference
                fill(lhs, 0);
                fill(rhs, 1);

                return reference::gemmlowp_matrix_multiply_core<int32_t, int8_t>(lhs, rhs, dst_shape, 0, 0);
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyReshaped3DValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
               bool interleave_lhs, bool interleave_rhs, DataType data_type)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = false;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = true;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, m_h, data_type);
        _reference = compute_reference(lhs_shape, rhs_shape, m_h, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
                std::uniform_int_distribution<> distribution(1, 254);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::QASYMM8_SIGNED:
            {
                std::uniform_int_distribution<> distribution(-127, 126);
                library->fill(tensor, distribution, i);
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, unsigned int m_h,
                              DataType data_type)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs_reshaped.info(), rhs_reshaped.info(), dst.info(), lhs_info, rhs_info, GEMMReshapeInfo(M, N, K, 1, 1, m_h));

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped }, { ACL_SRC_1, &rhs_reshaped }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, unsigned int m_h, DataType data_type)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        switch(data_type)
        {
            case DataType::QASYMM8:
            {
                // Create reference
                SimpleTensor<uint8_t> lhs{ lhs_shape, data_type, 1 };
                SimpleTensor<uint8_t> rhs{ rhs_shape, data_type, 1 };

                // Fill reference
                fill(lhs, 0);
                fill(rhs, 1);

                return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
            }
            case DataType::QASYMM8_SIGNED:
            {
                // Create reference
                SimpleTensor<int8_t> lhs{ lhs_shape, data_type, 1 };
                SimpleTensor<int8_t> rhs{ rhs_shape, data_type, 1 };

                // Fill reference
                fill(lhs, 0);
                fill(rhs, 1);

                return reference::gemmlowp_matrix_multiply_core<int32_t, int8_t>(lhs, rhs, dst_shape, 0, 0);
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename ReshapeRHSOperatorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyReshapedOnlyRHSValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0,
               unsigned int k0, unsigned int h0, bool interleave_rhs, bool transpose_rhs, DataType data_type)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = transpose_rhs;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
                std::uniform_int_distribution<> distribution(1, 254);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::QASYMM8_SIGNED:
            {
                std::uniform_int_distribution<> distribution(-127, 126);
                library->fill(tensor, distribution, i);
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info,
                              const GEMMRHSMatrixInfo &rhs_info, DataType data_type)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        GEMMKernelInfo gemm_info;
        gemm_info.m        = M;
        gemm_info.n        = N;
        gemm_info.k        = K;
        gemm_info.lhs_info = lhs_info;
        gemm_info.rhs_info = rhs_info;
        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs.info(), rhs_reshaped.info(), dst.info(), gemm_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &rhs_reshaped, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs }, { ACL_SRC_1, &rhs_reshaped }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        if(data_type == DataType::QASYMM8)
        {
            // Create reference
            SimpleTensor<uint8_t> lhs{ lhs_shape, data_type, 1 };
            SimpleTensor<uint8_t> rhs{ rhs_shape, data_type, 1 };

            // Fill reference
            fill(lhs, 0);
            fill(rhs, 1);

            return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
        }
        else
        {
            // Create reference
            SimpleTensor<int8_t> lhs{ lhs_shape, data_type, 1 };
            SimpleTensor<int8_t> rhs{ rhs_shape, data_type, 1 };

            // Fill reference
            fill(lhs, 0);
            fill(rhs, 1);

            return reference::gemmlowp_matrix_multiply_core<int32_t, int8_t>(lhs, rhs, dst_shape, 0, 0);
        }
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename T, typename TensorType, typename AccessorType, typename ReshapeRHSOperatorType, typename GEMMFunctionType, typename ReduceOperation, typename CastOperation>
class GEMMLowpMatrixMultiplyReshapedOnlyRHSMMULOutputStageValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0,
               unsigned int k0, unsigned int h0, bool interleave_rhs, bool transpose_rhs, bool broadcast_bias, DataType data_type)
    {
        GEMMLowpOutputStageInfo output_stage;
        output_stage.type                    = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_stage.output_data_type        = data_type;
        output_stage.gemmlowp_multipliers    = std::vector<int32_t> { 1 };
        output_stage.gemmlowp_shifts         = std::vector<int32_t> { 1 };
        output_stage.gemmlowp_multipliers[0] = 1;
        output_stage.gemmlowp_shifts[0]      = 1;
        output_stage.gemmlowp_offset         = 0;
        constexpr float scale                = 0.001f;
        quantization::calculate_quantized_multiplier(scale, &output_stage.gemmlowp_multipliers[0], &output_stage.gemmlowp_shifts[0]);
        output_stage.gemmlowp_min_bound = -100;
        output_stage.gemmlowp_max_bound = 100;

        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = transpose_rhs;

        int a_offset = 1;
        int b_offset = 1;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, output_stage, a_offset, b_offset);
        if(gemm_validated == true)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, bias_shape, data_type, output_stage, a_offset, b_offset);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
                std::uniform_int_distribution<> distribution(1, 254);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::QASYMM8_SIGNED:
            {
                std::uniform_int_distribution<> distribution(-127, 126);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::S32:
            {
                std::uniform_int_distribution<> distribution(-10000, 10000);
                library->fill(tensor, distribution, i);
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info,
                              const GEMMRHSMatrixInfo &rhs_info, DataType data_type, GEMMLowpOutputStageInfo output_stage, const int a_offset, const int b_offset)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1, QuantizationInfo(1.0f / 255, a_offset));
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1, QuantizationInfo(1.0f / 255, b_offset));
        TensorType bias = create_tensor<TensorType>(bias_shape, DataType::S32, 1);
        TensorType dst;
        TensorType rhs_reshaped;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // Tensors for precomputing sum of lhs rows / rhs columns
        TensorType vec_sum_rows = create_tensor<TensorType>(TensorShape(M, 1, lhs_shape[2]), DataType::S32, 1);
        TensorType vec_sum_cols = create_tensor<TensorType>(TensorShape(N, 1, rhs_shape[2]), DataType::S32, 1);

        GEMMKernelInfo gemm_info;
        gemm_info.m            = M;
        gemm_info.n            = N;
        gemm_info.k            = K;
        gemm_info.lhs_info     = lhs_info;
        gemm_info.rhs_info     = rhs_info;
        gemm_info.output_stage = output_stage;
        gemm_info.a_offset     = a_offset;
        gemm_info.b_offset     = b_offset;
        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);

        // If GEMM is not validated, do not try to run. The validation will check
        // if the technology supports this extension. If not, the test will be skipped.
        // If it supports, the test will fail anyway because target and reference
        // will not match.
        gemm_validated = bool(gemm.validate(lhs.info(), rhs_reshaped.info(), dst.info(), gemm_info, vec_sum_cols.info(), vec_sum_rows.info(), bias.info()));
        if(gemm_validated == true)
        {
            gemm.configure(lhs.info(), rhs_reshaped.info(), dst.info(), gemm_info, vec_sum_cols.info(), vec_sum_rows.info(), bias.info());

            ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

            // Allocate tensors
            lhs.allocator()->allocate();
            rhs.allocator()->allocate();
            rhs_reshaped.allocator()->allocate();
            bias.allocator()->allocate();
            vec_sum_cols.allocator()->allocate();
            vec_sum_rows.allocator()->allocate();
            dst.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!vec_sum_cols.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!vec_sum_rows.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

            // Fill tensors
            fill(AccessorType(lhs), 0);
            fill(AccessorType(rhs), 1);
            fill(AccessorType(bias), 2);

            TensorType    lhs_32 = create_tensor<TensorType>(lhs_shape, DataType::S32, 1);
            TensorType    rhs_32 = create_tensor<TensorType>(rhs_shape, DataType::S32, 1);
            CastOperation cast_lhs;
            CastOperation cast_rhs;
            cast_lhs.configure(&lhs, &lhs_32, ConvertPolicy::SATURATE);
            cast_rhs.configure(&rhs, &rhs_32, ConvertPolicy::SATURATE);
            lhs_32.allocator()->allocate();
            rhs_32.allocator()->allocate();
            cast_lhs.run();
            cast_rhs.run();

            ReduceOperation lhs_sum_rows;
            ReduceOperation rhs_sum_cols;

            lhs_sum_rows.configure(&lhs_32, &vec_sum_rows, 0, ReductionOperation::SUM, false);
            rhs_sum_cols.configure(&rhs_32, &vec_sum_cols, 1, ReductionOperation::SUM);

            lhs_sum_rows.run();
            rhs_sum_cols.run();

            // Compute GEMM
            ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
            reshape_rhs.run(reshape_rhs_pack);
            ITensorPack gemm_pack({ { ACL_SRC_0, &lhs }, { ACL_SRC_1, &rhs_reshaped }, { ACL_SRC_2, &bias }, { ACL_DST, &dst }, { ACL_VEC_COL_SUM, &vec_sum_cols }, { ACL_VEC_ROW_SUM, &vec_sum_rows } });
            gemm.run(gemm_pack);
        }

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, DataType data_type, GEMMLowpOutputStageInfo output_stage,
                                      const int a_offset, const int b_offset)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T>       lhs{ lhs_shape, data_type, 1, QuantizationInfo(1.0f / 255, a_offset) };
        SimpleTensor<T>       rhs{ rhs_shape, data_type, 1, QuantizationInfo(1.0f / 255, b_offset) };
        SimpleTensor<int32_t> bias{ bias_shape, DataType::S32, 1 };
        SimpleTensor<int32_t> dst{ dst_shape, DataType::S32, 1 };
        SimpleTensor<T>       dst_final{ dst_shape, data_type, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        dst       = reference::gemmlowp_matrix_multiply_core<int32_t, T>(lhs, rhs, dst_shape, a_offset, b_offset);
        dst_final = reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, T>(dst, bias,
                                                                                      output_stage.gemmlowp_multipliers, output_stage.gemmlowp_shifts, output_stage.gemmlowp_offset, output_stage.gemmlowp_min_bound, output_stage.gemmlowp_max_bound);
        return dst_final;
    }

    bool            gemm_validated = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename ReshapeRHSOperatorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyReshapedOnlyRHSMMULValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0,
               unsigned int k0, unsigned int h0, bool interleave_rhs, bool transpose_rhs, DataType data_type)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = transpose_rhs;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type);
        if(gemm_validated == true)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
                std::uniform_int_distribution<> distribution(1, 254);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::QASYMM8_SIGNED:
            {
                std::uniform_int_distribution<> distribution(-127, 126);
                library->fill(tensor, distribution, i);
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info,
                              const GEMMRHSMatrixInfo &rhs_info, DataType data_type)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        GEMMKernelInfo gemm_info;
        gemm_info.m        = M;
        gemm_info.n        = N;
        gemm_info.k        = K;
        gemm_info.lhs_info = lhs_info;
        gemm_info.rhs_info = rhs_info;
        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);

        // If GEMM is not validated, do not try to run. The validation will check
        // if the technology supports this extension. If not, the test will be skipped.
        // If it supports, the test will fail anyway because target and reference
        // will not match.
        gemm_validated = bool(gemm.validate(lhs.info(), rhs_reshaped.info(), dst.info(), gemm_info, nullptr, nullptr, nullptr));
        if(gemm_validated == true)
        {
            gemm.configure(lhs.info(), rhs_reshaped.info(), dst.info(), gemm_info, nullptr, nullptr, nullptr);

            ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

            // Allocate tensors
            lhs.allocator()->allocate();
            rhs.allocator()->allocate();
            rhs_reshaped.allocator()->allocate();
            dst.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

            // Fill tensors
            fill(AccessorType(lhs), 0);
            fill(AccessorType(rhs), 1);

            // Compute GEMM
            ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
            reshape_rhs.run(reshape_rhs_pack);
            ITensorPack gemm_pack({ { ACL_SRC_0, &lhs }, { ACL_SRC_1, &rhs_reshaped }, { ACL_DST, &dst } });
            gemm.run(gemm_pack);
        }

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        if(data_type == DataType::QASYMM8)
        {
            // Create reference
            SimpleTensor<uint8_t> lhs{ lhs_shape, data_type, 1 };
            SimpleTensor<uint8_t> rhs{ rhs_shape, data_type, 1 };
            SimpleTensor<int32_t> dst{ dst_shape, DataType::S32, 1 };

            // Fill reference
            fill(lhs, 0);
            fill(rhs, 1);

            return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
        }
        else
        {
            // Create reference
            SimpleTensor<int8_t>  lhs{ lhs_shape, data_type, 1 };
            SimpleTensor<int8_t>  rhs{ rhs_shape, data_type, 1 };
            SimpleTensor<int32_t> dst{ dst_shape, DataType::S32, 1 };

            // Fill reference
            fill(lhs, 0);
            fill(rhs, 1);

            return reference::gemmlowp_matrix_multiply_core<int32_t, int8_t>(lhs, rhs, dst_shape, 0, 0);
        }
    }

    bool                  gemm_validated = true;
    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename ReshapeRHSOperatorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyReshapedOnlyRHS3DValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0,
               unsigned int k0, unsigned int h0, bool interleave_rhs, bool transpose_rhs, DataType data_type)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = transpose_rhs;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, m_h, data_type);
        _reference = compute_reference(lhs_shape, rhs_shape, m_h, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
                std::uniform_int_distribution<> distribution(1, 254);
                library->fill(tensor, distribution, i);
            }
            break;
            case DataType::QASYMM8_SIGNED:
            {
                std::uniform_int_distribution<> distribution(-127, 126);
                library->fill(tensor, distribution, i);
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Unsupported data type");
        }
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info,
                              const GEMMRHSMatrixInfo &rhs_info, unsigned int m_h, DataType data_type)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        GEMMKernelInfo gemm_info;
        gemm_info.m                   = M;
        gemm_info.n                   = N;
        gemm_info.k                   = K;
        gemm_info.depth_output_gemm3d = m_h;
        gemm_info.lhs_info            = lhs_info;
        gemm_info.rhs_info            = rhs_info;
        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs.info(), rhs_reshaped.info(), dst.info(), gemm_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &rhs_reshaped, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs }, { ACL_SRC_1, &rhs_reshaped }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, unsigned int m_h, DataType data_type)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        if(data_type == DataType::QASYMM8)
        {
            // Create reference
            SimpleTensor<uint8_t> lhs{ lhs_shape, data_type, 1 };
            SimpleTensor<uint8_t> rhs{ rhs_shape, data_type, 1 };

            // Fill reference
            fill(lhs, 0);
            fill(rhs, 1);

            return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
        }
        else
        {
            // Create reference
            SimpleTensor<int8_t> lhs{ lhs_shape, data_type, 1 };
            SimpleTensor<int8_t> rhs{ rhs_shape, data_type, 1 };

            // Fill reference
            fill(lhs, 0);
            fill(rhs, 1);

            return reference::gemmlowp_matrix_multiply_core<int32_t, int8_t>(lhs, rhs, dst_shape, 0, 0);
        }
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyNativeValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0 = n0;
        rhs_info.k0 = k0;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info);
        _reference = compute_reference(lhs_shape, rhs_shape);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
        std::uniform_int_distribution<> distribution(1, 254);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, DataType::QASYMM8, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, DataType::QASYMM8, 1);
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        GEMMFunctionType gemm;
        gemm.configure(lhs.info(), rhs.info(), dst.info(), lhs_info, rhs_info, GEMMReshapeInfo(M, N, K));

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs }, { ACL_SRC_1, &rhs }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<uint8_t> lhs{ lhs_shape, DataType::QASYMM8, 1 };
        SimpleTensor<uint8_t> rhs{ rhs_shape, DataType::QASYMM8, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);

        return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename GEMMFunctionType>
class GEMMLowpMatrixMultiplyNative3DValidationFixture : public framework::Fixture
{
public:
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0 = n0;
        rhs_info.k0 = k0;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, m_h);
        _reference = compute_reference(lhs_shape, rhs_shape, m_h);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        // Between 1 and 254 in order to avoid having -128 and 128 for the DOT product path
        std::uniform_int_distribution<> distribution(1, 254);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, unsigned int m_h)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, DataType::QASYMM8, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, DataType::QASYMM8, 1);
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        GEMMFunctionType gemm;
        gemm.configure(lhs.info(), rhs.info(), dst.info(), lhs_info, rhs_info, GEMMReshapeInfo(M, N, K, 1, 1, m_h));

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs }, { ACL_SRC_1, &rhs }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, unsigned int m_h)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<uint8_t> lhs{ lhs_shape, DataType::QASYMM8, 1 };
        SimpleTensor<uint8_t> rhs{ rhs_shape, DataType::QASYMM8, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);

        return reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(lhs, rhs, dst_shape, 0, 0);
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_GEMMLOWPFIXTURE_H
