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

#ifndef CKW_PROTOTYPE_EXAMPLES_COMMON_EXAMPLEKERNELWRITER_H
#define CKW_PROTOTYPE_EXAMPLES_COMMON_EXAMPLEKERNELWRITER_H

#include "ckw/KernelWriter.h"
#include "ckw/TensorTileSampler.h"

class ExampleComponentArgument;

namespace ckw
{
class Kernel;
} // namespace ckw

/** Extended implementation of kernel writer for dynamic fusion. */
class ExampleKernelWriter : public ckw::KernelWriter
{
public:
    /** Initialize a new instance of @ref ExampleKernelWriter class.
     *
     * @param[in] kernel The kernel to be generated.
     */
    explicit ExampleKernelWriter(ckw::Kernel &kernel);

    /** Load the user tensor to the tile in the same component argument if it hasn't been loaded.
     *
     * @param[in] tensor_or_tile The component argument that is either a user tensor or a virtual tensor.
     * @param[in] sampler        The tensor sampling information to load the tile.
     */
    void op_load_once(ExampleComponentArgument *tensor_or_tile, const ckw::TensorTileSampler &sampler);
};

#endif // CKW_PROTOTYPE_EXAMPLES_COMMON_EXAMPLEKERNELWRITER_H
