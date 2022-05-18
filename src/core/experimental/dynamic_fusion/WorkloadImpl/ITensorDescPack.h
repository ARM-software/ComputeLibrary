/*
 * Copyright (c) 2022 Arm Limited.
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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_ITENSORDESCPACK_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_ITENSORDESCPACK_H

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
template <typename TDesc>
class ITensorDescPack
{
public:
    struct PackElement
    {
        PackElement()                    = default;
        ~PackElement()                   = default;
        PackElement(const PackElement &) = default;
        PackElement &operator=(const PackElement &) = default;
        PackElement(PackElement &&)                 = default;
        PackElement &operator=(PackElement &&) = default;
        PackElement(int id, TDesc *tensor)
            : id(id), tensor(tensor), ctensor(nullptr)
        {
        }
        PackElement(int id, const TDesc *ctensor)
            : id(id), tensor(nullptr), ctensor(ctensor)
        {
        }

        int          id{ -1 };
        TDesc       *tensor{ nullptr };
        const TDesc *ctensor{ nullptr };

        friend bool operator==(const PackElement &elem0, const PackElement &elem1)
        {
            const bool same_ctensor = (elem0.tensor == nullptr && elem1.tensor == nullptr && elem0.ctensor != nullptr && elem1.ctensor != nullptr && *elem0.ctensor == *elem1.ctensor);
            const bool same_tensor  = (elem0.ctensor == nullptr && elem1.ctensor == nullptr && elem0.tensor != nullptr && elem1.tensor != nullptr && *elem0.tensor == *elem1.tensor);

            return elem0.id == elem1.id && (same_ctensor || same_tensor);
        }
    };

public:
    /** Default Constructor */
    ITensorDescPack()                                           = default;
    ~ITensorDescPack()                                          = default;
    ITensorDescPack<TDesc>(const ITensorDescPack<TDesc> &other) = default;
    ITensorDescPack<TDesc> &operator=(const ITensorDescPack<TDesc> &other) = default;
    ITensorDescPack<TDesc>(ITensorDescPack<TDesc> &&other)                 = default;
    ITensorDescPack<TDesc> &operator=(ITensorDescPack<TDesc> &&other) = default;
    /**  Initializer list Constructor */
    ITensorDescPack(std::initializer_list<PackElement> l)
        : _pack{}
    {
        for(auto &e : l)
        {
            _pack[e.id] = e;
        }
    }
    /** Add tensor to the pack
     *
     * @param[in] id     ID/type of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_tensor(int id, TDesc *tensor)
    {
        _pack[id] = PackElement(id, tensor);
    }

    /** Add const tensor to the pack
     *
     * @param[in] id     ID/type of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_const_tensor(int id, const TDesc *tensor)
    {
        _pack[id] = PackElement(id, tensor);
    }
    /** Get tensor of a given id from the pac
     *
     * @param[in] id ID of tensor to extract
     *
     * @return The pointer to the tensor if exist and is non-const else nullptr
     */
    TDesc *get_tensor(int id)
    {
        auto it = _pack.find(id);
        return it != _pack.end() ? it->second.tensor : nullptr;
    }
    /** Get constant tensor of a given id
     *
     * @param[in] id ID of tensor to extract
     *
     * @return The pointer to the tensor if exist and is const else nullptr
     */
    const TDesc *get_const_tensor(int id) const
    {
        auto it = _pack.find(id);
        if(it != _pack.end())
        {
            return it->second.ctensor != nullptr ? it->second.ctensor : it->second.tensor;
        }
        return nullptr;
    }
    /** Remove the tensor stored with the given id
     *
     * @param[in] id ID of tensor to remove
     */
    void remove_tensor(int id)
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
     * @return std::vector<TDesc *>
     */
    std::vector<TDesc *> get_src_tensors()
    {
        std::vector<TDesc *> src_tensors{};
        for(int id = static_cast<int>(TensorType::ACL_SRC); id <= static_cast<int>(TensorType::ACL_SRC_END); ++id)
        {
            auto tensor = get_tensor(id);
            if(tensor != nullptr)
            {
                src_tensors.push_back(tensor);
            }
        }
        return src_tensors;
    }
    /** Get the const ACL_SRC_* tensors
     *
     * @return std::vector<const TDesc *>
     */
    std::vector<const TDesc *> get_const_src_tensors() const
    {
        std::vector<const TDesc *> src_tensors{};
        for(int id = static_cast<int>(TensorType::ACL_SRC); id <= static_cast<int>(TensorType::ACL_SRC_END); ++id)
        {
            auto tensor = get_const_tensor(id);
            if(tensor != nullptr)
            {
                src_tensors.push_back(tensor);
            }
        }
        return src_tensors;
    }
    /** Get the ACL_DST_* tensors
     *
     * @return std::vector<TDesc *>
     */
    std::vector<TDesc *> get_dst_tensors()
    {
        std::vector<TDesc *> dst_tensors{};
        for(int id = static_cast<int>(TensorType::ACL_DST); id <= static_cast<int>(TensorType::ACL_DST_END); ++id)
        {
            auto tensor = get_tensor(id);
            if(tensor != nullptr)
            {
                dst_tensors.push_back(tensor);
            }
        }
        return dst_tensors;
    }
    /** Get the const ACL_DST_* tensors
     *
     * @return std::vector<const TDesc *>
     */
    std::vector<const TDesc *> get_const_dst_tensors() const
    {
        std::vector<const TDesc *> dst_tensors{};
        for(int id = static_cast<int>(TensorType::ACL_DST); id <= static_cast<int>(TensorType::ACL_DST_END); ++id)
        {
            auto tensor = get_const_tensor(id);
            if(tensor != nullptr)
            {
                dst_tensors.push_back(tensor);
            }
        }
        return dst_tensors;
    }

    friend bool operator==(const ITensorDescPack<TDesc> &pack0, const ITensorDescPack<TDesc> &pack1)
    {
        return pack0._pack == pack1._pack;
    }

private:
    std::unordered_map<int, PackElement> _pack{}; /**< Container with the packed tensors */
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_ITENSORDESCPACK_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */