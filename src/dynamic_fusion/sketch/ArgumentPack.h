/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_ARGUMENTPACK_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_ARGUMENTPACK_H

#include "arm_compute/core/experimental/Types.h"

#include <unordered_map>
#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** This is a generic class that packs the arguments of an operator. For now, it is only used for tensor-related types
 * Examples of "tensor-related types": @ref ITensorInfo, @ref ITensor, @ref ICLTensor
 *
 * The argument id is the position of the argument within the pack, and is represented by @ref TensorType
 *
 * @tparam T Tensor-related type
 */
template <typename T>
class ArgumentPack
{
public:
    /** @ref arm_compute::TensorType encodes the position of a tensor argument within the pack */
    using Id = TensorType;
    /** A single argument element within the pack
     * It contains either a const pointer or a non-const pointer to the Tensor-related type T, but never at the same time
     */
    struct PackElement
    {
        PackElement()                                   = default;
        PackElement(const PackElement &elem)            = default;
        PackElement &operator=(const PackElement &elem) = default;
        PackElement(PackElement &&elem)                 = default;
        PackElement &operator=(PackElement &&elem)      = default;
        PackElement(Id id, T *tensor) : id(id), tensor(tensor), ctensor(nullptr)
        {
        }
        PackElement(Id id, const T *ctensor) : id(id), tensor(nullptr), ctensor(ctensor)
        {
        }

        Id       id{ACL_UNKNOWN};  /**< Argument id within the pack */
        T       *tensor{nullptr};  /**< Non-const pointer to tensor-related object */
        const T *ctensor{nullptr}; /**< Const pointer to tensor-related object */
    };

public:
    /** Default constructor */
    ArgumentPack() = default;
    /** Destructor */
    ~ArgumentPack() = default;
    /** Allow instances of this class to be copy constructed */
    ArgumentPack<T>(const ArgumentPack<T> &other) = default;
    /** Allow instances of this class to be copied */
    ArgumentPack<T> &operator=(const ArgumentPack<T> &other) = default;
    /** Allow instances of this class to be move constructed */
    ArgumentPack<T>(ArgumentPack<T> &&other) = default;
    /** Allow instances of this class to be moved */
    ArgumentPack<T> &operator=(ArgumentPack<T> &&other) = default;
    /** Initializer list Constructor */
    ArgumentPack(const std::initializer_list<PackElement> &l) : _pack{}
    {
        for (const auto &e : l)
        {
            _pack[e.id] = e;
        }
    }
    /** Add tensor to the pack
     *
     * @param[in] id     ID of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_tensor(Id id, T *tensor)
    {
        _pack[id] = PackElement(id, tensor);
    }
    /** Add const tensor to the pack
     *
     * @param[in] id     ID of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_const_tensor(Id id, const T *tensor)
    {
        _pack[id] = PackElement(id, tensor);
    }
    /** Get tensor of a given id from the pack
     *
     * @param[in] id ID of tensor to extract
     *
     * @return The pointer to the tensor if exist and is non-const else nullptr
     */
    T *get_tensor(Id id)
    {
        auto it = _pack.find(id);
        return it != _pack.end() ? it->second.tensor : nullptr;
    }
    /** Get constant tensor of a given id
     *
     * @param[in] id ID of tensor to extract
     *
     * @return The pointer to the tensor (const or not) if exist else nullptr
     */
    const T *get_const_tensor(Id id) const
    {
        auto it = _pack.find(id);
        if (it != _pack.end())
        {
            return it->second.ctensor != nullptr ? it->second.ctensor : it->second.tensor;
        }
        return nullptr;
    }
    /** Remove the tensor stored with the given id
     *
     * @param[in] id ID of tensor to remove
     */
    void remove_tensor(Id id)
    {
        _pack.erase(id);
    }
    /** Pack size accessor
     *
     * @return Number of tensors registered to the pack
     */
    size_t size() const
    {
        return _pack.size();
    }
    /** Checks if pack is empty
     *
     * @return True if empty else false
     */
    bool empty() const
    {
        return _pack.empty();
    }
    /** Get the ACL_SRC_* tensors
     *
     * @return std::vector<T *>
     */
    std::vector<T *> get_src_tensors()
    {
        std::vector<T *> src_tensors{};
        for (int id = static_cast<int>(TensorType::ACL_SRC); id <= static_cast<int>(TensorType::ACL_SRC_END); ++id)
        {
            auto tensor = get_tensor(static_cast<TensorType>(id));
            if (tensor != nullptr)
            {
                src_tensors.push_back(tensor);
            }
        }
        return src_tensors;
    }
    /** Get the const ACL_SRC_* tensors
     *
     * @return std::vector<const T *>
     */
    std::vector<const T *> get_const_src_tensors() const
    {
        std::vector<const T *> src_tensors{};
        for (int id = static_cast<int>(TensorType::ACL_SRC); id <= static_cast<int>(TensorType::ACL_SRC_END); ++id)
        {
            auto tensor = get_const_tensor(static_cast<TensorType>(id));
            if (tensor != nullptr)
            {
                src_tensors.push_back(tensor);
            }
        }
        return src_tensors;
    }
    /** Get the ACL_DST_* tensors
     *
     * @return std::vector<T *>
     */
    std::vector<T *> get_dst_tensors()
    {
        std::vector<T *> dst_tensors{};
        for (int id = static_cast<int>(TensorType::ACL_DST); id <= static_cast<int>(TensorType::ACL_DST_END); ++id)
        {
            auto tensor = get_tensor(static_cast<TensorType>(id));
            if (tensor != nullptr)
            {
                dst_tensors.push_back(tensor);
            }
        }
        return dst_tensors;
    }
    /** Get the const ACL_DST_* tensors
     *
     * @return std::vector<const T *>
     */
    std::vector<const T *> get_const_dst_tensors() const
    {
        std::vector<const T *> dst_tensors{};
        for (int id = static_cast<int>(TensorType::ACL_DST); id <= static_cast<int>(TensorType::ACL_DST_END); ++id)
        {
            auto tensor = get_const_tensor(static_cast<TensorType>(id));
            if (tensor != nullptr)
            {
                dst_tensors.push_back(tensor);
            }
        }
        return dst_tensors;
    }

private:
    std::unordered_map<int, PackElement> _pack{}; /**< Container with the packed tensors */
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_ARGUMENTPACK_H
