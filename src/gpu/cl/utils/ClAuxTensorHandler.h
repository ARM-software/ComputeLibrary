/*
 * Copyright (c) 2021, 2024 Arm Limited.
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
#ifndef ACL_SRC_GPU_CL_UTILS_CLAUXTENSORHANDLER_H
#define ACL_SRC_GPU_CL_UTILS_CLAUXTENSORHANDLER_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include "src/common/utils/Log.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
/** Tensor handler to wrap and handle tensor allocations on workspace buffers
 *
 * @note About memory handling using bypass_* flags
 *       See @ref arm_compute::cpu::CpuAuxTensorHandler
 */
class CLAuxTensorHandler
{
public:
    /** Create a temporary tensor handle, by either important an existing tensor from a tensor pack, or allocating a
     *  new one.
     *
     * @param[in]     slot_id       Slot id of the tensor to be retrieved in the tensor pack
     *                              If no such tensor exists in the tensor pack, a new tensor will be allocated.
     * @param[in]     info          Tensor info containing requested size of the new tensor.
     *                              If requested size is larger than the tensor retrieved from the tensor pack,
     *                              a new tensor will be allocated.
     * @param[in,out] pack          Tensor pack to retrieve the old tensor. When @p pack_inject is true, the new
     *                              tensor will also be added here.
     * @param[in]     pack_inject   In case of a newly allocated tensor, whether to add this tensor back to the
     *                              @p pack
     * @param[in]     bypass_alloc  Bypass allocation in case of a new tensor
     *                              This is to prevent unnecessary memory operations when the handler object is not
     *                              used
     * @param[in]     bypass_import Bypass importation in case of a retrieved tensor
     *                                  This is to prevent unnecessary memory operations when the handler object is not
     *                                  used
     */
    CLAuxTensorHandler(int          slot_id,
                       TensorInfo  &info,
                       ITensorPack &pack,
                       bool         pack_inject   = false,
                       bool         bypass_alloc  = false,
                       bool         bypass_import = false)
        : _tensor()
    {
        if (info.total_size() == 0)
        {
            return;
        }
        _tensor.allocator()->soft_init(info);

        ICLTensor *packed_tensor = utils::cast::polymorphic_downcast<ICLTensor *>(pack.get_tensor(slot_id));
        if ((packed_tensor == nullptr) || (info.total_size() > packed_tensor->info()->total_size()))
        {
            if (!bypass_alloc)
            {
                _tensor.allocator()->allocate();
                ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Allocating auxiliary tensor");
            }

            if (pack_inject)
            {
                pack.add_tensor(slot_id, &_tensor);
                _injected_tensor_pack = &pack;
                _injected_slot_id     = slot_id;
            }
        }
        else
        {
            if (!bypass_import)
            {
                _tensor.allocator()->import_memory(packed_tensor->cl_buffer());
            }
        }
    }

    /** Create a temporary handle to the original tensor with a new @ref TensorInfo
     * This is useful if we want to change a tensor's tensor info at run time without modifying the original tensor
     *
     * @param[in] info          New tensor info to "assign" to @p tensor
     * @param[in] tensor        Tensor to be assigned a new @ref TensorInfo
     * @param[in] bypass_import Bypass importing @p tensor's memory into the handler.
     *                          This is to prevent unnecessary memory operations when the handler object is not used
     */
    CLAuxTensorHandler(TensorInfo &info, ICLTensor &tensor, bool bypass_import = false) : _tensor()
    {
        _tensor.allocator()->soft_init(info);
        if (!bypass_import)
        {
            ARM_COMPUTE_ERROR_ON(tensor.info() == nullptr);
            if (info.total_size() <= tensor.info()->total_size())
            {
                _tensor.allocator()->import_memory(tensor.cl_buffer());
            }
        }
    }

    CLAuxTensorHandler(const CLAuxTensorHandler &)          = delete;
    CLAuxTensorHandler &operator=(const CLAuxTensorHandler) = delete;

    ~CLAuxTensorHandler()
    {
        if (_injected_tensor_pack)
        {
            _injected_tensor_pack->remove_tensor(_injected_slot_id);
        }
    }

    ICLTensor *get()
    {
        return &_tensor;
    }

    ICLTensor *operator()()
    {
        return &_tensor;
    }

private:
    CLTensor     _tensor{};
    ITensorPack *_injected_tensor_pack{nullptr};
    int          _injected_slot_id{TensorType::ACL_UNKNOWN};
};
} // namespace opencl
} // namespace arm_compute
#endif // ACL_SRC_GPU_CL_UTILS_CLAUXTENSORHANDLER_H
