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

#ifndef CKW_INCLUDE_CKW_TYPES_CONSTANTDATA_H
#define CKW_INCLUDE_CKW_TYPES_CONSTANTDATA_H

#include "ckw/Error.h"
#include "ckw/types/DataType.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace ckw
{
// Forward Declarations
class KernelWriter;

class ConstantData
{
    using String       = std::string;
    using StringVector = std::vector<String>;

public:
    /** Templated constructor */
    template <typename T>
    ConstantData(std::initializer_list<std::initializer_list<T>> values, DataType data_type);

    /** Templated constructor */
    template <typename T>
    ConstantData(const std::vector<std::vector<T>> &values, DataType data_type);

private:
    /** Validate the given data type and the template type
     *
     * @param[in] data_type data type
     *
     * @return true if user provided data type and the template type are conformant
     */
    template <typename T>
    bool validate(DataType data_type);

    /** Get the constant data as a 2d vector of string values
     *
     * @return a 2d vector of strings that has the string-converted values
     */
    const std::vector<StringVector> &values() const;

    /** Get the underlying data type of the constant values
     *
     * @return a @ref ckw::DataType object that represents the underlying data type
     */
    DataType data_type() const;

    // Friends
    friend class KernelWriter;

private:
    // Data members
    std::vector<StringVector> _values{};
    DataType                  _data_type{};
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TYPES_CONSTANTDATA_H
