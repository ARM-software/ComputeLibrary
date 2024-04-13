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

#ifndef CKW_SRC_CL_CLMEMORYOPBUFFERHELPER_H
#define CKW_SRC_CL_CLMEMORYOPBUFFERHELPER_H

#include "src/cl/helpers/ICLMemoryOpHelper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ckw
{

// Forward Declarations
class CLKernelWriter;
class CLTile;
enum class MemoryOperation;

/** Helper class to write memory operations (like load/store) in OpenCL
 */
class CLMemoryOpBufferHelper : public ICLMemoryOpHelper
{
public:
    /** Constructor similar to @ref ICLMemoryOpHelper() */
    CLMemoryOpBufferHelper(CLKernelWriter *writer, ITensor *tensor, TensorSampler *sampler, MemoryOperation op)
        : ICLMemoryOpHelper(writer, tensor, sampler, op)
    {
    }

    /** Copy constructor */
    CLMemoryOpBufferHelper(const CLMemoryOpBufferHelper &) = default;

    /** Assignment operator overload */
    CLMemoryOpBufferHelper &operator=(const CLMemoryOpBufferHelper &) = default;

    // Methods overridden
    void initialize(const CLTile *dst, const CLTile *x, const CLTile *z, const CLTile *b) override;
    void write_row(int32_t row_id, const std::string &coord_y) override;
    void finalize() override;

private:
    struct LeftoverDescriptor
    {
        LeftoverDescriptor(const std::string &dst, const std::string &coord, const std::string &statement)
            : dst(dst), coord(coord), statement(statement)
        {
        }

        std::string dst{};       // Describes the destination tile or part of it
        std::string coord{};     // Describes the coordinate to be used in boundary checks
        std::string statement{}; // Describes the memory operation statement
    };

    std::vector<int32_t>            _ls_width_part{};
    std::vector<LeftoverDescriptor> _leftovers_x{};
    std::string                     _coord_orig_z{};

    static bool validate(const CLKernelWriter *writer,
                         const ITensor        *tensor,
                         const TensorSampler  *sampler,
                         const Tensor3dMapper *mapper,
                         MemoryOperation       op,
                         const CLTile         *dst);

    void out_of_bound_initialize_x(const std::string &coord);
    void out_of_bound_finalize_x();
    void out_of_bound_initialize_y(const std::string &coord);
    void out_of_bound_finalize_y(const std::string &dst);
    void out_of_bound_initialize_z(const std::string &coord);
    void out_of_bound_finalize_z();

    std::string
    to_statement(MemoryOperation op, int32_t vector_width, const std::string &data, const std::string &address) const;
    std::string
    to_buffer_address(const std::string &x, const std::string &y, const std::string &z, const std::string &b) const;
};
} // namespace ckw

#endif /* CKW_SRC_CL_CLMEMORYOPBUFFERHELPER_H */
