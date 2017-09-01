/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_DATA_TYPE_DATASET_H__
#define __ARM_COMPUTE_TEST_DATA_TYPE_DATASET_H__

#include "arm_compute/core/Types.h"

#ifdef BOOST
#include "tests/validation_old/boost_wrapper.h"
#endif /* BOOST */

namespace arm_compute
{
namespace test
{
/** Abstract data set containing data types.
 *
 * Can be used as input for Boost data test cases to automatically run a test
 * case on different data types.
 */
template <unsigned int Size>
class DataTypes
{
public:
    /** Type of the samples in the data set. */
    using sample = DataType;

    /** Dimensionality of the data set. */
    enum
    {
        arity = 1
    };

    /** Number of samples in the data set. */
#ifdef BOOST
    boost::unit_test::data::size_t size() const
#else  /* BOOST */
    unsigned int size() const
#endif /* BOOST */
    {
        return _types.size();
    }

    /** Type of the iterator used to step through all samples in the data set.
     * Needs to support operator*() and operator++() which a pointer does.
     */
    using iterator = const DataType *;

    /** Iterator to the first sample in the data set. */
    iterator begin() const
    {
        return _types.data();
    }

protected:
    /** Protected constructor to make the class abstract. */
    template <typename... Ts>
    DataTypes(Ts &&... types)
        : _types{ { types... } }
    {
    }

    /** Protected destructor to prevent deletion of derived classes through a
     * pointer to the base class.
     */
    ~DataTypes() = default;

private:
    std::array<DataType, Size> _types;
};

/** Data set containing all data types. */
class AllDataTypes final : public DataTypes<14>
{
public:
    AllDataTypes()
        : DataTypes{ DataType::U8, DataType::S8, DataType::U16, DataType::S16,
                     DataType::U32, DataType::S32, DataType::U64, DataType::S64,
                     DataType::F16, DataType::F32, DataType::F64, DataType::SIZET,
                     DataType::QS8, DataType::QS16 }
    {
    }

    ~AllDataTypes() = default;
};

/** Data set containing all unsigned data types. */
class UnsignedDataTypes final : public DataTypes<4>
{
public:
    UnsignedDataTypes()
        : DataTypes{ DataType::U8, DataType::U16, DataType::U32, DataType::U64 }
    {
    }

    ~UnsignedDataTypes() = default;
};

/** Data set containing all signed data types. */
class SignedDataTypes final : public DataTypes<4>
{
public:
    SignedDataTypes()
        : DataTypes{ DataType::S8, DataType::S16, DataType::S32, DataType::S64 }
    {
    }

    ~SignedDataTypes() = default;
};

/** Data set containing all floating point data types. */
class FloatDataTypes final : public DataTypes<3>
{
public:
    FloatDataTypes()
        : DataTypes{ DataType::F16, DataType::F32, DataType::F64 }
    {
    }

    ~FloatDataTypes() = default;
};

/** Data set containing all fixed point data types. */
class FixedPointDataTypes final : public DataTypes<2>
{
public:
    FixedPointDataTypes()
        : DataTypes{ DataType::QS8, DataType::QS16 }
    {
    }

    ~FixedPointDataTypes() = default;
};

/** Supported CNN float types. */
class CNNFloatDataTypes final : public DataTypes<1>
{
public:
    CNNFloatDataTypes()
        : DataTypes{ DataType::F32 }
    {
    }

    ~CNNFloatDataTypes() = default;
};

/** Supported CNN fixed point types. */
class CNNFixedPointDataTypes final : public DataTypes<2>
{
public:
    CNNFixedPointDataTypes()
        : DataTypes{ DataType::QS8, DataType::QS16 }
    {
    }

    ~CNNFixedPointDataTypes() = default;
};

/** Supported CNN types. */
class CNNDataTypes final : public DataTypes<2>
{
public:
    CNNDataTypes()
        : DataTypes{ DataType::F32, DataType::QS8 }
    {
    }

    ~CNNDataTypes() = default;
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_DATA_TYPE_DATASET_H__ */
