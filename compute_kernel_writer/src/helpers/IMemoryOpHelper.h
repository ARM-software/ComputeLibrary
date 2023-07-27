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

#ifndef CKW_SRC_IMEMORYOPHELPER_H
#define CKW_SRC_IMEMORYOPHELPER_H

#include <cstdint>
#include <string>

namespace ckw
{

// Forward Declarations
class ITile;
class KernelWriter;
class Tensor3dMapper;
enum class MemoryOperation;

/** Base class for the backend specific helper classes
 *  that helps writing code for memory operations like load/store.
 */
class IMemoryOpHelper
{
public:
    /** Constructor
     *
     * @param[in] x      @ref ckw::KernelWriter object to write the code
     * @param[in] mapper @ref ckw::Tensor3dMapper object that tells how to map the Nd tensor to 3d
     * @param[in] op     The memory operation to be done (e.g. Load/Store)
     */
    IMemoryOpHelper(KernelWriter *x, Tensor3dMapper *mapper, MemoryOperation op)
        : _writer(x), _mapper(mapper), _op(op)
    {
    }

    /** Copy constructor */
    IMemoryOpHelper(const IMemoryOpHelper &) = default;

    /** Assignment operator overload */
    IMemoryOpHelper &operator=(const IMemoryOpHelper &) = default;

    /** Destructor */
    virtual ~IMemoryOpHelper() = default;

    /** Initialization method that takes a 3D tensor's x, z dimensions and
     *  the batch offset as a tile object, and initializes the code inside
     *  the writer object.
     *
     * @param[in] dst  tile object to perform the memory operation on
     * @param[in] x    tile object that describes the x-coordinate of the tensor involved
     * @param[in] z    tile object that describes the z-coordinate of the tensor involved
     * @param[in] b    tile object that describes the batch offset of the tensor involved
     */
    virtual void initialize(ITile *dst, ITile *x, ITile *z, ITile *b) = 0;

    /** Method that writes the actual code to the writer that performs the mentioned memory
     *  operation on the tile initialized. It writes the code for a specific row given in the
     *  arguments.
     *
     * @param[in] y   a pair where the elements are (row_id, y-coordinate string)
     */
    virtual void write(const std::pair<int32_t, std::string> &y) = 0;

    /** Method that finalizes the code in the writer object. This part is usually for taking
     *  care of finalizing anything that's been initialized inside @ref IMemoryHelper::initialize()
     *  such as matching compound statements, checking certain boundary conditions etc. No inputs
     *  and/or outputs, only the writer object is affected.
     */
    virtual void finalize() = 0;

protected:
    KernelWriter *_writer;
    Tensor3dMapper *_mapper;
    MemoryOperation  _op;
};
} // namespace ckw

#endif /* CKW_SRC_IMEMORYOPHELPER_H */
