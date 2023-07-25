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

#ifndef CKW_SRC_TENSOR3DMAPPER_H
#define CKW_SRC_TENSOR3DMAPPER_H

#include "ckw/TensorSampler.h"

#include <string>
namespace ckw
{

class ITensor;
class TensorSampler;

/** This internal-only class is responsible to map an Nd tensor to a 3d tensor with the
 *  help of TensorSampler object. The aim of the dimensionality reduction is to reduce
 *  the address calculation to:
 *      x + y * stride_y + z * stride_z + offset, where offset is determined by the batch.
 */
class Tensor3dMapper
{
public:
    /** Constructor */
    Tensor3dMapper(ITensor *tensor, TensorSampler sampler);

    /** Get dimension x as string */
    std::string tensor_component_x() const;

    /** Get dimension y as string */
    std::string tensor_component_y() const;

    /** Get dimension z as string */
    std::string tensor_component_z() const;

    /** Get stride for dimension x as string */
    std::string tensor_component_stride_x() const;

    /** Get stride for dimension y as string */
    std::string tensor_component_stride_y() const;

    /** Get stride for dimension z as string */
    std::string tensor_component_stride_z() const;

    /** Get stride for batch dimension as string */
    std::string tensor_component_stride_batch() const;

    /** Get the tensor sampler */
    TensorSampler sampler() const;

    /** Get the associated tensor */
    ITensor *tensor() const;

private:
    ITensor         *_tensor;
    TensorSampler    _sampler;
};
} // namespace ckw

#endif /* CKW_SRC_TENSOR3DMAPPER_H */
