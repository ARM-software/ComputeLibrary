/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_DATASET
#define ARM_COMPUTE_TEST_DATASET

#include <string>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace dataset
{
/** Abstract dataset base class. */
class Dataset
{
protected:
    /** Default constructor. */
    Dataset() = default;
    /** Default destructor. */
    ~Dataset() = default;

public:
    /** Allow instances of this class to be move constructed */
    Dataset(Dataset &&) = default;
};

/** Abstract implementation of a named dataset.
 *
 * The name should describe the values of the dataset.
 */
class NamedDataset : public Dataset
{
protected:
    /** Construct the dataset with the given name.
     *
     * @param[in] name Description of the values.
     */
    explicit NamedDataset(std::string name)
        : _name{ std::move(name) }
    {
    }

    /** Default destructor. */
    ~NamedDataset() = default;

public:
    /** Allow instances of this class to be move constructed */
    NamedDataset(NamedDataset &&) = default;

    /** Return name of the dataset.
     *
     * @return Description of the values.
     */
    std::string name() const
    {
        return _name;
    }

protected:
    const std::string _name;
};
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET */
