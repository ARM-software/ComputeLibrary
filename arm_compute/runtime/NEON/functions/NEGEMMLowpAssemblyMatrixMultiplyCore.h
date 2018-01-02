
/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMLOWPASSEMBLYMATRIXMULTIPLYCORE_H__
#define __ARM_COMPUTE_NEGEMMLOWPASSEMBLYMATRIXMULTIPLYCORE_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to execute matrix multiply assembly kernels.
 *
*/
class NEGEMMLowpAssemblyMatrixMultiplyCore : public IFunction
{
public:
    /** Constructor */
    NEGEMMLowpAssemblyMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  a      First input tensor  (Matrix A). Data type supported: U8, S8.
     * @param[in]  b      Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[out] output Output tensor. Data type supported: Data type supported: S32
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *output);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                _memory_group;
    std::unique_ptr<INEKernel> _mm_kernel;
    std::unique_ptr<INEKernel> _mtx_a_reshape_kernel;
    std::unique_ptr<INEKernel> _mtx_b_reshape_kernel;
    Tensor                     _tmp_a;
    Tensor                     _tmp_b;
    Tensor                     _workspace;
};
}
#endif /*__ARM_COMPUTE_NEGEMMLOWPASSEMBLYMATRIXMULTIPLYCORE_H__ */
