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

#ifndef CKW_SRC_CL_HELPERS_ICLMEMORYOPHELPER_H
#define CKW_SRC_CL_HELPERS_ICLMEMORYOPHELPER_H

#include "ckw/TensorSampler.h"

#include "src/Tensor3dMapper.h"
#include "src/TileView.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ckw
{

// Forward Declarations
class CLTile;
class CLKernelWriter;
class ITensor;
class TensorSampler;
enum class MemoryOperation;

/** Base class OpenCL memory operation helper classes
 *  that helps writing code for memory operations like load/store.
 */
class ICLMemoryOpHelper
{
public:
    /** Constructor
     *
     * @param[in] writer  @ref ckw::CLKernelWriter object to write the code
     * @param[in] tensor  @ref ckw::ITensor object to perform the memory operation on
     * @param[in] sampler @ref ckw::TensorSampler object that tells how to sample a tensor
     * @param[in] op      The memory operation to be done (e.g. Load/Store)
     * @param[in] dst     The tile to perform the memory operation on
     */
    ICLMemoryOpHelper(CLKernelWriter         *writer,
                      ITensor                *tensor,
                      TensorSampler          *sampler,
                      MemoryOperation         op,
                      const TileView<CLTile> &dst)
        : _writer(writer), _tensor(tensor), _sampler(sampler), _op(op), _dst(dst)
    {
        _mapper        = std::make_unique<Tensor3dMapper>(tensor, sampler->format());
        _ls_width_full = _dst.width();
    }

    /** Copy constructor */
    ICLMemoryOpHelper(const ICLMemoryOpHelper &) = delete;

    /** Assignment operator overload */
    ICLMemoryOpHelper &operator=(const ICLMemoryOpHelper &) = delete;

    /** Destructor */
    virtual ~ICLMemoryOpHelper() = default;

    /** Initialization method that takes a 3D tensor's x, z dimensions and
     *  the batch offset as a tile object, and initializes the code inside
     *  the writer object.
     *
     * @param[in] x    tile object that describes the x-coordinate of the tensor involved
     * @param[in] z    tile object that describes the z-coordinate of the tensor involved
     * @param[in] b    tile object that describes the batch offset of the tensor involved
     */
    virtual void initialize(const CLTile *x, const CLTile *z, const CLTile *b) = 0;

    /** Method that writes the actual code to the writer that performs the mentioned memory
     *  operation on the tile initialized. It writes the code for a specific row given in the
     *  arguments.
     *
     * @param[in] row_id   row id
     * @param[in] coord_y  y-coordinate as string
     */
    virtual void write_row(int32_t row_id, const std::string &coord_y) = 0;

    /** Method that finalizes the code in the writer object. This part is usually for taking
     *  care of finalizing anything that's been initialized inside @ref IMemoryHelper::initialize()
     *  such as matching compound statements, checking certain boundary conditions etc. No inputs
     *  and/or outputs, only the writer object is affected.
     */
    virtual void finalize() = 0;

protected:
    CLKernelWriter                 *_writer{nullptr};
    ITensor                        *_tensor{nullptr};
    TensorSampler                  *_sampler{nullptr};
    MemoryOperation                 _op;
    std::unique_ptr<Tensor3dMapper> _mapper{nullptr};
    TileView<CLTile>                _dst{};
    int32_t                         _ls_width_full{0};
    std::string                     _coord_x{};
    std::string                     _coord_z{};
    std::string                     _coord_b{};
};
} // namespace ckw

#endif // CKW_SRC_CL_HELPERS_ICLMEMORYOPHELPER_H
