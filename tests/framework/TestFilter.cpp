/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "TestFilter.h"

#include "Framework.h"
#include "support/ToolchainSupport.h"

#include <sstream>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
TestFilter::TestFilter(DatasetMode mode, const std::string &name_filter, const std::string &id_filter)
    : _dataset_mode{ mode }, _name_filter{ name_filter }, _id_filter{ parse_id_filter(id_filter) }
{
}

bool TestFilter::is_selected(const TestInfo &info) const
{
    const bool include_disabled = (info.mode == _dataset_mode) && (_dataset_mode == DatasetMode::DISABLED);
    if((info.mode & _dataset_mode) == DatasetMode::DISABLED && !include_disabled)
    {
        return false;
    }

    if(!std::regex_search(info.name, _name_filter))
    {
        return false;
    }

    if(!_id_filter.empty())
    {
        bool found = false;

        for(const auto range : _id_filter)
        {
            if(range.first <= info.id && info.id <= range.second)
            {
                found = true;
                break;
            }
        }

        if(!found)
        {
            return false;
        }
    }

    return true;
}

TestFilter::Ranges TestFilter::parse_id_filter(const std::string &id_filter) const
{
    Ranges      ranges;
    std::string str;
    bool        in_range = false;
    int         value    = 0;
    int         start    = 0;
    int         end      = std::numeric_limits<int>::max();

    std::stringstream stream(id_filter);

    // Get first value
    std::getline(stream, str, ',');

    if(stream.fail())
    {
        return ranges;
    }

    if(str.find("...") != std::string::npos)
    {
        in_range = true;
    }
    else
    {
        start = support::cpp11::stoi(str);
        end   = start;
    }

    while(!stream.eof())
    {
        std::getline(stream, str, ',');

        if(stream.fail())
        {
            break;
        }

        if(str.find("...") != std::string::npos)
        {
            end      = std::numeric_limits<int>::max();
            in_range = true;
        }
        else
        {
            value = support::cpp11::stoi(str);

            if(in_range || end == value - 1)
            {
                end      = value;
                in_range = false;
            }
            else
            {
                ranges.emplace_back(start, end);
                start = value;
                end   = start;
            }
        }
    }

    ranges.emplace_back(start, end);
    return ranges;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
