/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE
#define ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/GEMMLowp.h"
#include "tests/validation/Validation.h"

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
    switch(tensor.data_type())
    {
        case DataType::QSYMM8_PER_CHANNEL:
        {
            int min_bound = 128;
            int max_bound = -127;
            for(size_t j = 0; j < tensor.quantization_info().scale().size(); j++)
            {
                std::pair<int, int> bounds = get_symm_quantized_per_channel_bounds(tensor.quantization_info(), -1.0f, 1.0f, i);
                if(bounds.first < min_bound)
                {
                    min_bound = bounds.first;
                }
                if(bounds.second > max_bound)
                {
                    max_bound = bounds.second;
                }
            }
            std::uniform_int_distribution<int32_t> distribution(min_bound, max_bound);
            library->fill(tensor, distribution, i);
            break;
        }
        case DataType::QASYMM8:
        {
            std::uniform_int_distribution<uint32_t> distribution(1, 254);
            library->fill(tensor, distribution, i);
            break;
        }
        case DataType::F16:
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
            library->fill(tensor, distribution, i);
            break;
        }
        case DataType::F32:
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
            break;
        }
        default:
            library->fill_tensor_uniform(tensor, i);
    }
}

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d, bool reinterpret_output_as_3d, typename OutputType, bool is_fused = false>
TensorType compute_gemmlowp_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, int32_t a_offset, int32_t b_offset,
                                   GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo(), DataType data_type_a = DataType::QASYMM8, DataType data_type_b = DataType::QASYMM8,
                                   QuantizationInfo b_qinfo = QuantizationInfo(), bool reshape_b_only_on_first_run = false)
{
    // Create tensors
    DataType data_type_output = output_stage.type == GEMMLowpOutputStageType::NONE ? DataType::S32 : data_type_a;

    TensorType a      = create_tensor<TensorType>(shape_a, data_type_a, 1);
    TensorType b      = create_tensor<TensorType>(shape_b, data_type_b, 1); // gemm output before output stage mismatch if i pass data_layout_output here. to be investigated
    TensorType output = create_tensor<TensorType>(shape_output, data_type_output, 1);

    a.info()->set_quantization_info(QuantizationInfo(1.0f / 255, a_offset));

    if(data_type_b == DataType::QSYMM8_PER_CHANNEL)
    {
        b.info()->set_quantization_info(b_qinfo);
    }
    else
    {
        b.info()->set_quantization_info(QuantizationInfo(1.0f / 255, b_offset));
    }
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
    fill(AccessorType(a), 0);
    fill(AccessorType(b), 1);

    if(is_fused)
    {
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        bias.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        fill(AccessorType(bias), 2);
    }
    // Compute GEMM function
    gemmlowp.run();
    return output;
}

template <bool reinterpret_input_as_3d, typename TI = uint8_t, typename TW = uint8_t, bool pretranspose_A = false, bool pretranspose_B = false>
SimpleTensor<int32_t> compute_gemmlowp_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, int32_t a_offset, int32_t b_offset,
                                                 DataType data_type_a = DataType::QASYMM8, DataType data_type_b = DataType::QASYMM8, QuantizationInfo b_qinfo = QuantizationInfo())
{
    TensorShape shape_a_to_use = shape_a;
    if(reinterpret_input_as_3d)
    {
        // Collapse the second and third dimension if the input is 3D
        shape_a_to_use.collapse(2U, 1U);
    }

    // Create reference
    SimpleTensor<TI> a{ shape_a_to_use, data_type_a, 1 };
    SimpleTensor<TW> b{ shape_b, data_type_b, 1, data_type_b == DataType::QSYMM8_PER_CHANNEL ? b_qinfo : QuantizationInfo(1.0f / 255, b_offset) };

    TensorShape shape_a_to_use_transposed{ shape_a_to_use };
    TensorShape shape_b_transposed{ shape_b };

    shape_a_to_use_transposed.set(0, shape_a_to_use[1]);
    shape_a_to_use_transposed.set(1, shape_a_to_use[0]);
    shape_b_transposed.set(0, shape_b[1]);
    shape_b_transposed.set(1, shape_b[0]);

    SimpleTensor<TI> a_transposed{ shape_a_to_use_transposed, data_type_a, 1 };
    SimpleTensor<TW> b_transposed{ shape_b_transposed, data_type_b, 1, data_type_b == DataType::QSYMM8_PER_CHANNEL ? b_qinfo : QuantizationInfo(1.0f / 255, b_offset) };

    // Fill reference
    fill(a, 0);
    fill(b, 1);

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

    return reference::gemmlowp_matrix_multiply_core<int32_t, TI, TW>((pretranspose_A ? a_transposed : a), (pretranspose_B ? b_transposed : b), shape_output, a_offset, b_offset);
}
}

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false>
class GEMMLowpMatrixMultiplyCoreValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset)
    {
        _target    = compute_target(shape_a, shape_b, shape_output, a_offset, b_offset);
        _reference = compute_reference(shape_a, shape_b, shape_output, a_offset, b_offset);
    }

protected:
    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, int32_t a_offset, int32_t b_offset)
    {
        return compute_gemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, int32_t>(shape_a, shape_b, shape_output, a_offset, b_offset);
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, int32_t a_offset, int32_t b_offset)
    {
        return compute_gemmlowp_reference<reinterpret_input_as_3d>(shape_a, shape_b, shape_output, a_offset, b_offset);
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, typename TI = uint8_t, typename TW = uint8_t>
class GEMMLowpMatrixMultiplyCoreFusedOffsetOutputGenericValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset, GEMMLowpOutputStageInfo output_stage, DataType data_type_b,
               bool reshape_b_only_on_first_run)
    {
        ARM_COMPUTE_ASSERT(output_stage.type != GEMMLowpOutputStageType::NONE);
        DataType data_type_a = data_type_b == DataType::QASYMM8_SIGNED ? DataType::QASYMM8_SIGNED : DataType::QASYMM8;

        if(data_type_b == DataType::QSYMM8_PER_CHANNEL)
        {
            output_stage.is_quantized_per_channel              = true;
            const size_t                          num_channels = shape_b[0];
            std::vector<float>                    scales(num_channels);
            std::uniform_real_distribution<float> distribution(0.f, 1.f);
            library->fill(scales, distribution, 0);
            output_stage.gemmlowp_multipliers.resize(num_channels);
            output_stage.gemmlowp_shifts.resize(num_channels);
            for(size_t i = 0; i < num_channels; ++i)
            {
                quantization::calculate_quantized_multiplier(scales[i], &output_stage.gemmlowp_multipliers[i], &output_stage.gemmlowp_shifts[i]);
            }

            _reference = compute_reference(shape_a, shape_b, shape_output, a_offset, 0, output_stage, data_type_a, data_type_b, QuantizationInfo(scales));
            _target    = compute_target(shape_a, shape_b, shape_output, a_offset, 0, output_stage, data_type_a, data_type_b, QuantizationInfo(scales), reshape_b_only_on_first_run);
        }
        else
        {
            _reference = compute_reference(shape_a, shape_b, shape_output, a_offset, b_offset, output_stage, data_type_a, data_type_b, QuantizationInfo());
            _target    = compute_target(shape_a, shape_b, shape_output, a_offset, b_offset, output_stage, data_type_a, data_type_b, QuantizationInfo(), reshape_b_only_on_first_run);
        }
    }

protected:
    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, int32_t a_offset, int32_t b_offset, GEMMLowpOutputStageInfo output_stage,
                              DataType data_type_a, DataType data_type_b, QuantizationInfo b_qinfo, bool reshape_b_only_on_first_run = false)
    {
        return compute_gemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, qasymm8_t, true>(shape_a, shape_b, shape_output, a_offset, b_offset,
                output_stage, data_type_a, data_type_b, b_qinfo, reshape_b_only_on_first_run);
    }

    SimpleTensor<TI> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, int32_t a_offset, int32_t b_offset,
                                       GEMMLowpOutputStageInfo output_stage, DataType data_type_a, DataType data_type_b, QuantizationInfo b_qinfo)
    {
        SimpleTensor<int32_t> output = compute_gemmlowp_reference<reinterpret_input_as_3d, TI, TW>(shape_a, shape_b, shape_output, a_offset, b_offset, data_type_a, data_type_b, b_qinfo);

        TensorShape           bias_shape(shape_b[0]);
        SimpleTensor<int32_t> bias{ bias_shape, DataType::S32, 1 };
        fill(bias, 2);

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
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset, GEMMLowpOutputStageInfo output_stage, DataType data_type_b)
    {
        GEMMLowpMatrixMultiplyCoreFusedOffsetOutputGenericValidationFixture<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, TI, TW>::setup(shape_a, shape_b,
                shape_output, a_offset, b_offset, output_stage, data_type_b, false);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpQuantizeDownInt32ToUint8ScaleValidationFixture : public framework::Fixture
{
public:
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, output_stage, a_offset, b_offset);
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
    template <typename...>
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
#endif /* ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE */
