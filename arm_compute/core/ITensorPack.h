/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ITENSORPACK_H
#define ARM_COMPUTE_ITENSORPACK_H

#include "arm_compute/core/experimental/Types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace arm_compute
{
// Forward declaration
class ITensor;

/** Tensor packing service */
class ITensorPack
{
public:
    struct PackElement
    {
        PackElement() = default;
        PackElement(int id, ITensor *tensor)
            : id(id), tensor(tensor), ctensor(nullptr)
        {
        }
        PackElement(int id, const ITensor *ctensor)
            : id(id), tensor(nullptr), ctensor(ctensor)
        {
        }

        int            id{ -1 };
        ITensor       *tensor{ nullptr };
        const ITensor *ctensor{ nullptr };
    };

public:
    /** Default Constructor */
    ITensorPack() = default;
    /**  Initializer list Constructor */
    ITensorPack(std::initializer_list<PackElement> l);
    /** Add tensor to the pack
     *
     * @param[in] id     ID/type of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_tensor(int id, ITensor *tensor);

    /** Add const tensor to the pack
     *
     * @param[in] id     ID/type of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_tensor(int id, const ITensor *tensor);

    /** Add const tensor to the pack
     *
     * @param[in] id     ID/type of the tensor to add
     * @param[in] tensor Tensor to add
     */
    void add_const_tensor(int id, const ITensor *tensor);
    /** Get tensor of a given id from the pac
     *
     * @param[in] id ID of tensor to extract
     *
     * @return The pointer to the tensor if exist and is non-const else nullptr
     */
    ITensor *get_tensor(int id);
    /** Get constant tensor of a given id
     *
     * @param[in] id ID of tensor to extract
     *
     * @return The pointer to the tensor if exist and is const else nullptr
     */
    const ITensor *get_const_tensor(int id) const;
    /** Remove the tensor stored with the given id
     *
     * @param[in] id ID of tensor to remove
     */
    void remove_tensor(int id);
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

private:
    std::unordered_map<int, PackElement> _pack{}; /**< Container with the packed tensors */
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_ITENSORPACK_H */
