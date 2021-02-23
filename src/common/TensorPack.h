/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_ITENSORPACK_H_
#define SRC_COMMON_ITENSORPACK_H_

#include "arm_compute/core/ITensorPack.h"
#include "src/common/IContext.h"

struct AclTensorPack_
{
    arm_compute::detail::Header header{ arm_compute::detail::ObjectType::TensorPack, nullptr };

protected:
    AclTensorPack_()  = default;
    ~AclTensorPack_() = default;
};

namespace arm_compute
{
// Forward declaration
class ITensor;
class ITensorV2;

/** Tensor packing service
 *
 * Class is responsible for creating and managing a collection of tensors.
 * Tensor packs can be passed to operators to be part of the mutable data of the execution.
 */
class TensorPack : public AclTensorPack_
{
public:
    /** Constructor
     *
     * @param[in] ctx Context to be used
     */
    explicit TensorPack(IContext *ctx);
    /** Destructor */
    ~TensorPack();
    /** Add tensor to the pack
     *
     * @param[in] tensor  Tensor to add
     * @param[in] slot_id Slot identification in respect to the operator of the tensor to add
     *
     * @return Status code
     */
    AclStatus add_tensor(ITensorV2 *tensor, int32_t slot_id);
    /** Pack size accessor
     *
     * @return Number of tensors registered to the pack
     */
    size_t size() const;
    /** Checks if pack is empty
     *
     * @return True if empty else false
     */
    bool empty() const;
    /** Checks if an object is valid
     *
     * @return True if valid else false
     */
    bool is_valid() const;
    /** Get tensor of a given id from the pac
     *
     * @param[in] slot_id Slot identification of tensor to extract
     *
     * @return The pointer to the tensor if exist and is non-const else nullptr
     */
    arm_compute::ITensor *get_tensor(int32_t slot_id);
    /** Get legacy tensor pack
     *
     * @return Legacy tensor pack
     */
    arm_compute::ITensorPack &get_tensor_pack();

private:
    arm_compute::ITensorPack _pack; /**< Pack that currently redirects to the existing TensorPack */
};

/** Extract internal representation of a TensoPack
 *
 * @param[in] pack Opaque tensor pack pointer
 *
 * @return The internal representation as an TensorPack
 */
inline TensorPack *get_internal(AclTensorPack pack)
{
    return static_cast<TensorPack *>(pack);
}

namespace detail
{
/** Check if an internal TensorPack is valid
 *
 * @param[in] pack Internal tensor pack to check
 *
 * @return A status code
 */
inline StatusCode validate_internal_pack(const TensorPack *pack)
{
    if(pack == nullptr || !pack->is_valid())
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[TensorPack]: Invalid tensor pack object");
        return StatusCode::InvalidArgument;
    }
    return StatusCode::Success;
}
} // namespace detail
} // namespace arm_compute
#endif /* SRC_COMMON_ITENSORPACK_H_ */
