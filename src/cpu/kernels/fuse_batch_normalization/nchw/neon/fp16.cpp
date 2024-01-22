/*
 * Copyright (c) 2023 Arm Limited.
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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/CpuTypes.h"
#include "src/cpu/kernels/fuse_batch_normalization/generic/impl.h"

namespace arm_compute
{
namespace cpu
{
void fp16_batch_normalization_nchw_non_fused(const Window       &window,
                                             ITensor            *input,
                                             ITensor            *output,
                                             const ITensor      *mean,
                                             const ITensor      *var,
                                             const ITensor      *beta,
                                             const ITensor      *gamma,
                                             float               epsilon,
                                             ActivationLayerInfo act_info)
{
    batch_normalization_nchw<float16_t, false, detail::dummy<float16_t, 8>>(window, input, output, mean, var, beta,
                                                                            gamma, epsilon, act_info);
}

void fp16_batch_normalization_nchw_non_fused_relu(const Window       &window,
                                                  ITensor            *input,
                                                  ITensor            *output,
                                                  const ITensor      *mean,
                                                  const ITensor      *var,
                                                  const ITensor      *beta,
                                                  const ITensor      *gamma,
                                                  float               epsilon,
                                                  ActivationLayerInfo act_info)
{
    batch_normalization_nchw<float16_t, true, detail::relu<float16_t, 8>>(window, input, output, mean, var, beta, gamma,
                                                                          epsilon, act_info);
}

void fp16_batch_normalization_nchw_non_fused_brelu(const Window       &window,
                                                   ITensor            *input,
                                                   ITensor            *output,
                                                   const ITensor      *mean,
                                                   const ITensor      *var,
                                                   const ITensor      *beta,
                                                   const ITensor      *gamma,
                                                   float               epsilon,
                                                   ActivationLayerInfo act_info)
{
    batch_normalization_nchw<float16_t, true, detail::brelu<float16_t, 8>>(window, input, output, mean, var, beta,
                                                                           gamma, epsilon, act_info);
}

void fp16_batch_normalization_nchw_non_fused_lubrelu(const Window       &window,
                                                     ITensor            *input,
                                                     ITensor            *output,
                                                     const ITensor      *mean,
                                                     const ITensor      *var,
                                                     const ITensor      *beta,
                                                     const ITensor      *gamma,
                                                     float               epsilon,
                                                     ActivationLayerInfo act_info)
{
    batch_normalization_nchw<float16_t, true, detail::lubrelu<float16_t, 8>>(window, input, output, mean, var, beta,
                                                                             gamma, epsilon, act_info);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
