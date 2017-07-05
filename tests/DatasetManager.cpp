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
#include "DatasetManager.h"

#include <map>

namespace arm_compute
{
namespace test
{
DatasetManager &DatasetManager::get()
{
    static DatasetManager instance;
    return instance;
}

void DatasetManager::set_mode(DatasetMode mode)
{
    _mode = mode;
}

DatasetManager::ShapesDataset DatasetManager::shapesDataset() const
{
    static const std::string                           name = "Shape";
    static const std::vector<arm_compute::TensorShape> shapes{ arm_compute::TensorShape(1U), arm_compute::TensorShape(2U), arm_compute::TensorShape(3U), arm_compute::TensorShape(10U), arm_compute::TensorShape(20U), arm_compute::TensorShape(30U) };

    switch(_mode)
    {
        case DatasetManager::DatasetMode::PRECOMMIT:
            return framework::dataset::make(name, shapes.cbegin(), shapes.cbegin() + 3);
            break;
        case DatasetManager::DatasetMode::NIGHTLY:
            return framework::dataset::make(name, shapes.cbegin() + 3, shapes.cend());
            break;
        case DatasetManager::DatasetMode::ALL:
        // Fallthrough
        default:
            return framework::dataset::make(name, shapes.cbegin(), shapes.cend());
    }
}

DatasetManager::DatasetMode dataset_mode_from_name(const std::string &name)
{
    static const std::map<std::string, DatasetManager::DatasetMode> modes =
    {
        { "all", DatasetManager::DatasetMode::ALL },
        { "precommit", DatasetManager::DatasetMode::PRECOMMIT },
        { "nightly", DatasetManager::DatasetMode::NIGHTLY },
    };

    try
    {
        return modes.at(name);
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
}
} // namespace test
} // namespace arm_compute
