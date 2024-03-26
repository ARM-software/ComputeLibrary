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

#ifndef CKW_SRC_ITENSORARGUMENT_H
#define CKW_SRC_ITENSORARGUMENT_H

#include "ckw/TensorInfo.h"
#include "ckw/types/TensorComponentType.h"
#include "ckw/types/TensorStorageType.h"

#include "src/ITile.h"

#include <string>
#include <vector>

namespace ckw
{

class ITensorComponent;

/** Tensor storage variable */
struct TensorStorageVariable
{
    std::string       val{""};                          /** Tensor storage as a string */
    TensorStorageType type{TensorStorageType::Unknown}; /** Tensor storage type */
};

/** Tensor argument base class.
 *  A tensor is a multidimensional array used to store data. To access an element (or multiple elements) from a tensor,
 *  the following information are required:
 *  -# The data memory object. For example, the pointer to the array
 *  -# The tensor components, such as the size of each tensor dimension, or the number of elements in bytes contained in each dimension (also known as the "stride")
 */
class ITensorArgument
{
public:
    virtual ~ITensorArgument() = default;
    /** Method to get the name of the tensor argument.
     *
     * @return the name of the tensor argument
     */
    std::string name() const
    {
        return _basename;
    }

    /** Method to get the tensor info
     *
     * @return the @ref TensorInfo
     */
    TensorInfo &info()
    {
        return _info;
    }

    /** Method to get the tensor info
     *
     * @return the @ref TensorInfo
     */
    const TensorInfo &info() const
    {
        return _info;
    }

protected:
    TensorInfo  _info{};       // Tensor info
    std::string _basename{""}; // Tensor name
};

/** Tensor component argument base class */
class ITensorComponentAccess
{
public:
    virtual ~ITensorComponentAccess() = default;
    /** Method to get the tensor component variable as a tile.
     *
     * @param[in] x The tensor component to query
     *
     * @return the tensor component variable as a @ref ITile.
     */
    virtual ITile &component(TensorComponentType x) = 0;
    /** Method to get all tensor components needed to access the data in the tensor
     *
     * The tensor components returned by this method must be all passed as kernel argument
     *
     * @return a vector containing all the tensor components as pointers to @ref ITensorComponent objects.
     */
    virtual std::vector<const ITensorComponent *> components() const = 0;
};

/** Tensor storage argument base class */
class ITensorStorageAccess
{
public:
    virtual ~ITensorStorageAccess() = default;
    /** Method to get the tensor storage as a string
     *
     * @param[in] x The tensor storage to query
     *
     * @return the tensor storage as a @ref TensorStorageVariable
     */
    virtual TensorStorageVariable &storage(TensorStorageType x) = 0;
    /** Method to get all tensor storages needed to access the data in the tensor
     *
     * The tensor storages returned by this method must be all passed as kernel argument
     *
     * @return a vector containing all the tensor storages as @ref TensorStorageVariable objects
     */
    virtual std::vector<TensorStorageVariable> storages() const = 0;
};

} // namespace ckw

#endif // CKW_SRC_ITENSORARGUMENT_H
