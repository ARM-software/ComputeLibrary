/*
 * Copyright (c) 2024 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMCONV2DFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMCONV2DFIXTURE_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/graph/Utils.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{

template <typename TensorType, typename AccessorType, typename FunctionType>
class CpuGemmConv2dValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape   input_shape,
               TensorShape   weights_shape,
               TensorShape   bias_shape,
               TensorShape   output_shape,
               PadStrideInfo info,
               Size2D        dilation)
    {
        _dilation = dilation;
        _hash     = input_shape[0] + input_shape[1] + input_shape[2] + input_shape[3] + weights_shape[0] +
                weights_shape[1] + weights_shape[2] + weights_shape[3];
        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(TensorShape          input_shape,
                              TensorShape          weights_shape,
                              const TensorShape   &bias_shape,
                              TensorShape          output_shape,
                              const PadStrideInfo &info)
    {
        // We need to permute to the same layout that the reference impl needs.
        permute(input_shape, PermutationVector(2U, 0U, 1U));
        permute(weights_shape, PermutationVector(2U, 0U, 1U));
        permute(output_shape, PermutationVector(2U, 0U, 1U));

        const auto src_info     = TensorInfo(input_shape, 1, DataType::F32, _data_layout);
        const auto weights_info = TensorInfo(weights_shape, 1, DataType::F32, _data_layout);
        const auto biases_info  = TensorInfo(bias_shape, 1, DataType::F32, _data_layout);
        auto       dst_info     = TensorInfo(output_shape, 1, DataType::F32, _data_layout);

        auto conv = std::make_unique<FunctionType>();
        conv->configure(&src_info, &weights_info, &biases_info, &dst_info, info);
        ARM_COMPUTE_ASSERT(conv->validate(&src_info, &weights_info, &biases_info, &dst_info, info));

        // Create tensors
        auto src     = create_tensor<Tensor>(src_info);
        auto weights = create_tensor<Tensor>(weights_info);
        auto biases  = create_tensor<Tensor>(biases_info);
        auto dst     = create_tensor<Tensor>(dst_info);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();

        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &src},
                             {arm_compute::TensorType::ACL_SRC_1, &weights},
                             {arm_compute::TensorType::ACL_SRC_2, &biases},
                             {arm_compute::TensorType::ACL_DST, &dst}};
        ITensorPack prep_pack{{arm_compute::TensorType::ACL_SRC_1, &weights},
                              {arm_compute::TensorType::ACL_SRC_2, &biases}};

        auto const aux_mem_req = conv->workspace();
        auto       mg          = MemoryGroup{};
        auto       ws          = manage_workspace<Tensor>(aux_mem_req, mg, run_pack, prep_pack);

        // Fill tensors
        fill(AccessorType(src), 0 + _hash);
        fill(AccessorType(weights), 1 + _hash);
        fill(AccessorType(biases), 2 + _hash);

        conv->prepare(prep_pack);
        conv->run(run_pack);

        src.allocator()->free();
        weights.allocator()->free();
        biases.allocator()->free();

        return dst;
    }

    SimpleTensor<float> compute_reference(const TensorShape   &input_shape,
                                          const TensorShape   &weights_shape,
                                          const TensorShape   &bias_shape,
                                          const TensorShape   &output_shape,
                                          const PadStrideInfo &info)
    {
        // Create reference
        SimpleTensor<float> src{input_shape, DataType::F32};
        SimpleTensor<float> weights{weights_shape, DataType::F32};
        SimpleTensor<float> bias{bias_shape, DataType::F32};

        fill(src, 0 + _hash);
        fill(weights, 1 + _hash);
        fill(bias, 2 + _hash);

        return reference::convolution_layer<float>(src, weights, bias, output_shape, info, _dilation);
    }

    TensorType          _target{};
    SimpleTensor<float> _reference{};
    Size2D              _dilation{};
    int32_t             _hash{0};
    DataLayout          _data_layout{DataLayout::NHWC};
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMCONV2DFIXTURE_H
