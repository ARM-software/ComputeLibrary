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

#ifndef CKW_SRC_CL_HELPERS_CLMEMORYOPIMAGE2DHELPER_H
#define CKW_SRC_CL_HELPERS_CLMEMORYOPIMAGE2DHELPER_H

#include "src/cl/helpers/ICLMemoryOpHelper.h"

#include <string>

namespace ckw
{

// Forward Declarations
class CLKernelWriter;
class CLTile;
enum class MemoryOperation;

/** Helper class to write memory operations (like load/store) in OpenCL for Image2d type */
class CLMemoryOpImage2dHelper : public ICLMemoryOpHelper
{
public:
    /** Constructor similar to @ref ICLMemoryOpHelper() */
    CLMemoryOpImage2dHelper(CLKernelWriter *writer, ITensor *tensor, TensorSampler *sampler, MemoryOperation op)
        : ICLMemoryOpHelper(writer, tensor, sampler, op)
    {
    }

    /** Copy constructor */
    CLMemoryOpImage2dHelper(const CLMemoryOpImage2dHelper &) = default;

    /** Assignment operator overload */
    CLMemoryOpImage2dHelper &operator=(const CLMemoryOpImage2dHelper &) = default;

    // Methods overridden
    void initialize(const CLTile *dst, const CLTile *x, const CLTile *z, const CLTile *b) override;
    void write_row(int32_t row_id, const std::string &coord_y) override;
    void finalize() override;

private:
    static bool validate(const CLKernelWriter *writer, const ITensor *tensor, const TensorSampler *sampler, const Tensor3dMapper *mapper, MemoryOperation op, const CLTile *dst);

    void out_of_bound_initialize_y(const std::string &coord);
    void out_of_bound_finalize_y();

    std::string to_ls_image2d(MemoryOperation op, int32_t vector_width, const std::string &data, const std::string &sampler, const std::string &address) const;
    std::string to_ls_image2d_sampler() const;
    std::string to_ls_image2d_address(const std::string &x, const std::string &y, const std::string &z, const std::string &b) const;
};
} // namespace ckw

#endif // CKW_SRC_CL_HELPERS_CLMEMORYOPIMAGE2DHELPER_H
