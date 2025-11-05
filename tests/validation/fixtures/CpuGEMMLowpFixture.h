/*
 * Copyright (c) 2017-2025 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMLOWPFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMLOWPFIXTURE_H

#include "tests/validation/fixtures/GEMMLowpFixture.h"

#include <cstdint>

namespace arm_compute
{
namespace test
{
namespace validation
{

namespace
{

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          bool reinterpret_input_as_3d,
          bool reinterpret_output_as_3d,
          typename OutputType,
          bool is_fused  = false,
          bool run_twice = false>
TensorType compute_cpugemmlowp_target(const TensorShape      &shape_a,
                                      const TensorShape      &shape_b,
                                      const TensorShape      &shape_output,
                                      const QuantizationInfo &a_qinfo,
                                      const QuantizationInfo &b_qinfo,
                                      const QuantizationInfo &output_qinfo,
                                      DataType                data_type_a                 = DataType::QASYMM8,
                                      DataType                data_type_b                 = DataType::QASYMM8,
                                      GEMMLowpOutputStageInfo output_stage                = GEMMLowpOutputStageInfo(),
                                      bool                    reshape_b_only_on_first_run = false,
                                      const TensorFillInfo   &finfo                       = TensorFillInfo(),
                                      bool                    accumulate                  = false,
                                      bool                    dynamic_qinfo               = false,
                                      DataType                data_type_output            = DataType::UNKNOWN)
{
    ARM_COMPUTE_ASSERT(is_data_type_quantized_asymmetric(data_type_a));

    // If unknown, set to sensible defaults
    if (data_type_output == DataType::UNKNOWN)
    {
        data_type_output = output_stage.type == GEMMLowpOutputStageType::NONE ? DataType::S32 : data_type_a;
    }

    // Create tensors
    TensorType a =
        create_tensor<TensorType>(shape_a, data_type_a, 1, dynamic_qinfo ? QuantizationInfo(1.0, 0, true) : a_qinfo);
    TensorType b = create_tensor<TensorType>(
        shape_b, data_type_b, 1,
        dynamic_qinfo
            ? QuantizationInfo(1.0, 0, true)
            : b_qinfo); // gemm output before output stage mismatch if i pass data_layout_output here. to be investigated
    TensorType output =
        create_tensor<TensorType>(shape_output, data_type_output, 1,
                                  output_qinfo /* output_qinfo will be ignored when output stage type is None */);
    TensorType bias;

    if (is_fused)
    {
        TensorShape bias_shape(shape_b[0]);
        bias =
            create_tensor<TensorType>(bias_shape, data_type_output == DataType::F32 ? DataType::F32 : DataType::S32, 1);
    }

    // Create and configure function
    // The GEMMinfo includes the values of the depth in case of reinterpreted 3d input/output
    FunctionType gemmlowp;
    gemmlowp.configure(a.info(), b.info(), is_fused ? bias.info() : nullptr, output.info(),
                       GEMMInfo(false, false, reshape_b_only_on_first_run,
                                (reinterpret_output_as_3d ? shape_output[2] : 0), reinterpret_input_as_3d, false,
                                output_stage, false /*fp_mixed_precision*/, false /*fast_math*/,
                                false /*broadcast_bias*/, arm_compute::ActivationLayerInfo(), false /* fixed_format */,
                                arm_compute::WeightFormat::UNSPECIFIED, false /* pretranspose_B */, accumulate));

    // If the QuantizationInfo is dynamic, it needs to be settable after configure (note that we also force it to be dynamic)
    if (dynamic_qinfo)
    {
        a.info()->set_quantization_info(QuantizationInfo(a_qinfo.scale(), a_qinfo.offset(), true));
        b.info()->set_quantization_info(QuantizationInfo(b_qinfo.scale(), b_qinfo.offset(), true));
        output.info()->set_quantization_info(QuantizationInfo(output_qinfo.scale(), output_qinfo.offset(), true));
        gemmlowp.update_quantization_parameters(a.info()->quantization_info(), b.info()->quantization_info(),
                                                output.info()->quantization_info(), data_type_output, true, true);
    }

    ARM_COMPUTE_ASSERT(a.info()->is_resizable());
    ARM_COMPUTE_ASSERT(b.info()->is_resizable());
    ARM_COMPUTE_ASSERT(output.info()->is_resizable());

    add_padding_x({&a, &b, &output});

    // Allocate tensors
    a.allocator()->allocate();
    b.allocator()->allocate();
    output.allocator()->allocate();

    ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
    ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
    ARM_COMPUTE_ASSERT(!output.info()->is_resizable());

    ITensorPack pack = {{arm_compute::TensorType::ACL_SRC_0, &a},
                        {arm_compute::TensorType::ACL_SRC_1, &b},
                        {arm_compute::TensorType::ACL_DST, &output}};

    // Fill tensors
    fill_quantized(AccessorType(a), 0 + finfo.hash);
    fill_quantized(AccessorType(b), 1 + finfo.hash);

    if (accumulate)
    {
        ARM_COMPUTE_ASSERT(accumulate != run_twice);
        fill(AccessorType(output), 6 + finfo.hash, finfo.min_output, finfo.max_output);
    }

    if (is_fused)
    {
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        bias.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        fill(AccessorType(bias), 2 + finfo.hash, finfo.min_bias, finfo.max_bias);
        pack.add_tensor(arm_compute::TensorType::ACL_SRC_2, &bias);
    }

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(gemmlowp.workspace(), mg, pack, pack);

    // Run with variable inputs.
    if (run_twice)
    {
        gemmlowp.run(pack);
        fill_quantized(AccessorType(a), 3 + finfo.hash); // Fill tensors with new seed after run
        fill_quantized(AccessorType(b), 4 + finfo.hash);

        if (is_fused)
        {
            fill(AccessorType(bias), 5 + finfo.hash, finfo.min_bias, finfo.max_bias);
        }
    }

    // Compute GEMM function
    gemmlowp.run(pack);

    return output;
}
} // namespace

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          bool reinterpret_input_as_3d  = false,
          bool reinterpret_output_as_3d = false,
          bool run_twice                = false>
class CpuGEMMLowpMatrixMultiplyCoreValidationFixture
    : protected GEMMLowpGenericMatrixMultiplyCoreValidationFixture<TensorType,
                                                                   AccessorType,
                                                                   FunctionType,
                                                                   reinterpret_input_as_3d,
                                                                   reinterpret_output_as_3d,
                                                                   run_twice>
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset)
    {
        const auto     a_qinfo = QuantizationInfo(1.0f / 255, a_offset);
        const auto     b_qinfo = QuantizationInfo(2.0f / 255, b_offset);
        TensorFillInfo finfo;

        bool accumulate    = false;
        bool dynamic_qinfo = false;
        this->_target =
            compute_target(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, accumulate, dynamic_qinfo);
        this->_reference = this->compute_reference(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, accumulate);
    }

protected:
    TensorType compute_target(const TensorShape      &shape_a,
                              const TensorShape      &shape_b,
                              const TensorShape      &shape_output,
                              const QuantizationInfo &a_qinfo,
                              const QuantizationInfo &b_qinfo,
                              const TensorFillInfo   &finfo,
                              const bool              accumulate,
                              const bool              dynamic_qinfo)
    {
        const auto output_qinfo = QuantizationInfo(); // No output stage
        return compute_cpugemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d,
                                          reinterpret_output_as_3d, int32_t, false, run_twice>(
            shape_a, shape_b, shape_output, a_qinfo, b_qinfo, output_qinfo, DataType::QASYMM8, DataType::QASYMM8,
            GEMMLowpOutputStageInfo(), false, finfo, accumulate, dynamic_qinfo, DataType::UNKNOWN);
    }
};

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          bool reinterpret_input_as_3d  = false,
          bool reinterpret_output_as_3d = false,
          bool run_twice                = false>
class CpuGEMMLowpStaticQuantMatrixMultiplyCoreValidationFixture
    : protected CpuGEMMLowpMatrixMultiplyCoreValidationFixture<TensorType,
                                                               AccessorType,
                                                               FunctionType,
                                                               reinterpret_input_as_3d,
                                                               reinterpret_output_as_3d,
                                                               run_twice>
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape shape_output,
               int32_t     a_offset,
               int32_t     b_offset,
               DataType    data_type)
    {
        ARM_COMPUTE_ASSERT(data_type == DataType::QASYMM8_SIGNED || data_type == DataType::QASYMM8);
        const auto     a_qinfo = QuantizationInfo(1.0f / 255, a_offset);
        const auto     b_qinfo = QuantizationInfo(2.0f / 255, b_offset);
        TensorFillInfo finfo;

        bool accumulate    = false;
        bool dynamic_qinfo = true;
        this->_target      = compute_target(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, accumulate,
                                            dynamic_qinfo, data_type);
        this->_reference   = compute_reference(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, data_type);
    }

protected:
    TensorType compute_target(const TensorShape      &shape_a,
                              const TensorShape      &shape_b,
                              const TensorShape      &shape_output,
                              const QuantizationInfo &a_qinfo,
                              const QuantizationInfo &b_qinfo,
                              const TensorFillInfo   &finfo,
                              const bool              accumulate,
                              const bool              dynamic_qinfo,
                              const DataType          data_type)
    {
        const auto output_qinfo = QuantizationInfo(a_qinfo.scale(), a_qinfo.offset()); // No output stage
        return compute_cpugemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d,
                                          reinterpret_output_as_3d, int32_t, false, run_twice>(
            shape_a, shape_b, shape_output, a_qinfo, b_qinfo, output_qinfo, data_type, data_type,
            GEMMLowpOutputStageInfo(), false, finfo, accumulate, dynamic_qinfo, DataType::UNKNOWN);
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape      &shape_a,
                                            const TensorShape      &shape_b,
                                            const TensorShape      &shape_output,
                                            const QuantizationInfo &a_qinfo,
                                            const QuantizationInfo &b_qinfo,
                                            const TensorFillInfo   &finfo,
                                            const DataType          data_type)
    {
        if (data_type == DataType::QASYMM8)
        {
            return compute_gemmlowp_reference<reinterpret_input_as_3d, uint8_t, uint8_t, false, false, run_twice>(
                shape_a, shape_b, shape_output, a_qinfo, b_qinfo, data_type, data_type, finfo);
        }
        else
        {
            return compute_gemmlowp_reference<reinterpret_input_as_3d, int8_t, int8_t, false, false, run_twice>(
                shape_a, shape_b, shape_output, a_qinfo, b_qinfo, data_type, data_type, finfo);
        }
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMLOWPFIXTURE_H
