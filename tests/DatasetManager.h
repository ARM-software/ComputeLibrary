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
#ifndef ARM_COMPUTE_TEST_DATASETMANAGER
#define ARM_COMPUTE_TEST_DATASETMANAGER

#include "arm_compute/core/TensorShape.h"
#include "framework/datasets/Datasets.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace arm_compute
{
namespace test
{
class DatasetManager final
{
public:
    enum class DatasetMode : unsigned int
    {
        ALL       = 0,
        PRECOMMIT = 1,
        NIGHTLY   = 2
    };

    using ShapesDataset = framework::dataset::RangeDataset<std::vector<arm_compute::TensorShape>::const_iterator>;

    static DatasetManager &get();

    void set_mode(DatasetMode mode);

    ShapesDataset shapesDataset() const;

private:
    DatasetManager()  = default;
    ~DatasetManager() = default;

    DatasetMode _mode{ DatasetMode::ALL };
};

DatasetManager::DatasetMode dataset_mode_from_name(const std::string &name);

inline ::std::stringstream &operator>>(::std::stringstream &stream, DatasetManager::DatasetMode &mode)
{
    std::string value;
    stream >> value;
    mode = dataset_mode_from_name(value);
    return stream;
}

inline ::std::stringstream &operator<<(::std::stringstream &stream, DatasetManager::DatasetMode mode)
{
    switch(mode)
    {
        case DatasetManager::DatasetMode::PRECOMMIT:
            stream << "PRECOMMIT";
            break;
        case DatasetManager::DatasetMode::NIGHTLY:
            stream << "NIGHTLY";
            break;
        case DatasetManager::DatasetMode::ALL:
            stream << "ALL";
            break;
        default:
            throw std::invalid_argument("Unsupported dataset mode");
    }

    return stream;
}

inline std::string to_string(const DatasetManager::DatasetMode &mode)
{
    std::stringstream stream;
    stream << mode;
    return stream.str();
}
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASETMANAGER */
