/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_NEPOOLING3DLAYER_H
#define ARM_COMPUTE_NEPOOLING3DLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;
class IMemoryManager;
/** Basic function to simulate a pooling 3d layer with the specified pooling operation. This function calls the following kernels:
 *
 * -# @ref cpu::CpuPool3d
 */
class NEPooling3dLayer : public IFunction
{
public:
    /** Constructor */
    NEPooling3dLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPooling3dLayer(const NEPooling3dLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPooling3dLayer &operator=(const NEPooling3dLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEPooling3dLayer(NEPooling3dLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEPooling3dLayer &operator=(NEPooling3dLayer &&) = delete;
    /** Default destructor */
    ~NEPooling3dLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NDHWC
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     *
     * @note Source tensor is padded with -inf for MAX pooling and 0 otherwise
     *
     * @param[in]  input     Source tensor. Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[out] output    Destination tensor.
     * @param[in]  pool_info Contains pooling operation information described in @ref Pooling3dLayerInfo.
     */
    void configure(const ITensor *input, ITensor *output, const Pooling3dLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPooling3dLayer
     *
     *
     * @param[in] input     Source tensor info. Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[in] output    Destination tensor info.
     * @param[in] pool_info Contains pooling operation information described in @ref Pooling3dLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Pooling3dLayerInfo &pool_info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
#endif /* ARM_COMPUTE_NEPOOLING3DLAYER_H */
