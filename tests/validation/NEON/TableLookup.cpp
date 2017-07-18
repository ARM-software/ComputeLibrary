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
#include "AssetsLibrary.h"
#include "Globals.h"
#include "NEON/Accessor.h"
#include "NEON/Helper.h"
#include "NEON/LutAccessor.h"
#include "PaddingCalculator.h"
#include "RawLutAccessor.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Helpers.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NETableLookup.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
/** Compute Table Lookup function.
 *
 * @param[in] shape     Shape of the input tensors
 * @param[in] data_type Datatype of the input/output tensors
 * @param[in] lut       The input LUT.
 *
 * @return Computed output tensor.
 */
Tensor compute_table_lookup(const TensorShape &shape, DataType data_type, Lut &lut)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    // Create and configure function
    NETableLookup table_lookup;
    table_lookup.configure(&src, &lut, &dst);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(Accessor(src), 0);

    // Compute function
    table_lookup.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(TableLookup)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, (SmallShapes() + LargeShapes()) * boost::unit_test::data::make({ DataType::U8, DataType::S16 }),
                     shape, data_type)
{
    //Create Lut
    const int num_elem = (data_type == DataType::U8) ? std::numeric_limits<uint8_t>::max() + 1 : std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::lowest() + 1;
    Lut       lut(num_elem, data_type);

    if(data_type == DataType::U8)
    {
        fill_lookuptable(LutAccessor<uint8_t>(lut));
    }
    else
    {
        fill_lookuptable(LutAccessor<int16_t>(lut));
    }

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NETableLookup table_lookup;
    table_lookup.configure(&src, &lut, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }),
                     shape, data_type)
{
    //Create Lut
    const int num_elem = (data_type == DataType::U8) ? std::numeric_limits<uint8_t>::max() + 1 : std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::lowest() + 1;
    Lut       lut(num_elem, data_type);

    if(data_type == DataType::U8)
    {
        //Create rawLut
        std::map<uint8_t, uint8_t> rawlut;

        //Fill the Lut
        fill_lookuptable(LutAccessor<uint8_t>(lut));
        fill_lookuptable(RawLutAccessor<uint8_t>(rawlut));

        // Compute function
        Tensor dst = compute_table_lookup(shape, data_type, lut);

        // Compute reference
        RawTensor ref_dst = Reference::compute_reference_table_lookup(shape, data_type, rawlut);

        // Validate output
        validate(Accessor(dst), ref_dst);
    }
    else
    {
        //Create rawLut
        std::map<int16_t, int16_t> rawlut;

        //Fill the Lut
        fill_lookuptable(LutAccessor<int16_t>(lut));
        fill_lookuptable(RawLutAccessor<int16_t>(rawlut));

        // Compute function
        Tensor dst = compute_table_lookup(shape, data_type, lut);

        // Compute reference
        RawTensor ref_dst = Reference::compute_reference_table_lookup(shape, data_type, rawlut);

        // Validate output
        validate(Accessor(dst), ref_dst);
    }
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge,
                     LargeShapes() * boost::unit_test::data::make({ DataType::U8, DataType::S16 }),
                     shape, data_type)
{
    //Create Lut
    const int num_elem = (data_type == DataType::U8) ? std::numeric_limits<uint8_t>::max() + 1 : std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::lowest() + 1;
    Lut       lut(num_elem, data_type);

    if(data_type == DataType::U8)
    {
        //Create rawLut
        std::map<uint8_t, uint8_t> rawlut;

        //Fill the Lut
        fill_lookuptable(LutAccessor<uint8_t>(lut));
        fill_lookuptable(RawLutAccessor<uint8_t>(rawlut));

        // Compute function
        Tensor dst = compute_table_lookup(shape, data_type, lut);

        // Compute reference
        RawTensor ref_dst = Reference::compute_reference_table_lookup(shape, data_type, rawlut);

        // Validate output
        validate(Accessor(dst), ref_dst);
    }
    else
    {
        //Create rawLut
        std::map<int16_t, int16_t> rawlut;

        //Fill the Lut
        fill_lookuptable(LutAccessor<int16_t>(lut));
        fill_lookuptable(RawLutAccessor<int16_t>(rawlut));

        // Compute function
        Tensor dst = compute_table_lookup(shape, data_type, lut);

        // Compute reference
        RawTensor ref_dst = Reference::compute_reference_table_lookup(shape, data_type, rawlut);

        // Validate output
        validate(Accessor(dst), ref_dst);
    }
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
