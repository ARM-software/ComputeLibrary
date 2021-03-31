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
#ifndef SRC_COMMON_ITENSOR_H_
#define SRC_COMMON_ITENSOR_H_

#include "src/common/IContext.h"
#include "src/common/utils/Validate.h"

struct AclTensor_
{
    arm_compute::detail::Header header{ arm_compute::detail::ObjectType::Tensor, nullptr };

protected:
    AclTensor_()  = default;
    ~AclTensor_() = default;
};

namespace arm_compute
{
// Forward declaration
class ITensor;

/** Base class specifying the tensor interface */
class ITensorV2 : public AclTensor_
{
public:
    /** Explict Operator Constructor
     *
     * @param[in] ctx Context to be used by the operator
     */
    explicit ITensorV2(IContext *ctx)
        : AclTensor_()
    {
        ARM_COMPUTE_ASSERT_NOT_NULLPTR(ctx);
        this->header.ctx = ctx;
        this->header.ctx->inc_ref();
    }
    /** Destructor */
    virtual ~ITensorV2()
    {
        this->header.ctx->dec_ref();
        this->header.type = detail::ObjectType::Invalid;
    };
    /** Checks if a queue is valid
     *
     * @return True if successful otherwise false
     */
    bool is_valid() const
    {
        return this->header.type == detail::ObjectType::Tensor;
    };
    /** Map tensor to a host pointer
     *
     * @return A pointer to the underlying backing memory if successful else nullptr
     */
    virtual void *map() = 0;
    /** Unmap tensor
     *
     * @return AclStatus A status cod
     */
    virtual StatusCode unmap() = 0;
    /** Import external memory handle
     *
     * @param[in] handle Memory to import
     * @param[in] type   Type of imported memory
     *
     * @return Status code
     */
    virtual StatusCode import(void *handle, ImportMemoryType type) = 0;
    /** Get the legacy tensor object
     *
     * @return The legacy underlying tensor object
     */
    virtual arm_compute::ITensor *tensor() const = 0;
    /** Get the size of the tensor in byte
     *
     * @note The size isn't based on allocated memory, but based on information in its descriptor (dimensions, data type, etc.).
     *
     * @return The size of the tensor in byte
     */
    size_t get_size() const;
    /** Get the descriptor of this tensor
     *
     * @return The descriptor describing the characteristics of this tensor
     */
    AclTensorDescriptor get_descriptor() const;
};

/** Extract internal representation of a Tensor
 *
 * @param[in] tensor Opaque tensor pointer
 *
 * @return The internal representation as an ITensor
 */
inline ITensorV2 *get_internal(AclTensor tensor)
{
    return static_cast<ITensorV2 *>(tensor);
}

namespace detail
{
/** Check if an internal tensor is valid
 *
 * @param[in] tensor Internal tensor to check
 *
 * @return A status code
 */
inline StatusCode validate_internal_tensor(const ITensorV2 *tensor)
{
    if(tensor == nullptr || !tensor->is_valid())
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[ITensorV2]: Invalid tensor object");
        return StatusCode::InvalidArgument;
    }
    return StatusCode::Success;
}
} // namespace detail
} // namespace arm_compute
#endif /* SRC_COMMON_ITENSOR_H_ */
