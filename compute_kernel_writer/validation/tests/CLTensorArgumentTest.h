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

#ifndef CKW_TESTS_CLTENSORARGUMENTTEST_H
#define CKW_TESTS_CLTENSORARGUMENTTEST_H

#include "common/Common.h"
#include "src/cl/CLHelpers.h"
#include "src/cl/CLTensorArgument.h"

#include <string>
#include <vector>

namespace ckw
{
class CLTensorArgumentComponentNamesTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { 12, 14, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentComponentNamesTest()
    {
        _components.push_back(TensorComponentType::Dim0);
        _components.push_back(TensorComponentType::Dim1);
        _components.push_back(TensorComponentType::Dim2);
        _components.push_back(TensorComponentType::Dim3);
        _components.push_back(TensorComponentType::Dim4);
        _components.push_back(TensorComponentType::Dim1xDim2);
        _components.push_back(TensorComponentType::Dim2xDim3);
        _components.push_back(TensorComponentType::OffsetFirstElement);
        _components.push_back(TensorComponentType::Stride0);
        _components.push_back(TensorComponentType::Stride1);
        _components.push_back(TensorComponentType::Stride2);
        _components.push_back(TensorComponentType::Stride3);
        _components.push_back(TensorComponentType::Stride4);

        _expected_vars.push_back("src_dim0");
        _expected_vars.push_back("src_dim1");
        _expected_vars.push_back("src_dim2");
        _expected_vars.push_back("src_dim3");
        _expected_vars.push_back("src_dim4");
        _expected_vars.push_back("src_dim1xdim2");
        _expected_vars.push_back("src_dim2xdim3");
        _expected_vars.push_back("src_offset_first_element");
        _expected_vars.push_back("src_stride0");
        _expected_vars.push_back("src_stride1");
        _expected_vars.push_back("src_stride2");
        _expected_vars.push_back("src_stride3");
        _expected_vars.push_back("src_stride4");
    }

    bool run() override
    {
        VALIDATE_ON_MSG(_components.size() == _expected_vars.size(), "The number of components and variables does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_tests = _expected_vars.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_tests; ++i)
        {
            CLTensorArgument arg(tensor_name, info, false /* return_dims_by_value */);

            const std::string expected_var_name = _expected_vars[i];
            const std::string actual_var_name   = arg.component(_components[i]).str;

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentVariableNamesTest";
    }

private:
    std::vector<TensorComponentType> _components{};
    std::vector<std::string>         _expected_vars{};
};

class CLTensorArgumentStorageNamesTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { 12, 14, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentStorageNamesTest()
    {
        _storages.push_back(TensorStorageType::BufferUint8Ptr);
        _storages.push_back(TensorStorageType::Texture2dReadOnly);
        _storages.push_back(TensorStorageType::Texture2dWriteOnly);

        _expected_vars.push_back("src_ptr");
        _expected_vars.push_back("src_img2d");
        _expected_vars.push_back("src_img2d");
    }

    bool run() override
    {
        VALIDATE_ON_MSG(_storages.size() == _expected_vars.size(), "The number of storages and variables does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_tests = _expected_vars.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_tests; ++i)
        {
            CLTensorArgument arg(tensor_name, info, false /* return_dims_by_value */);

            const std::string expected_var_name = _expected_vars[i];
            const std::string actual_var_name   = arg.storage(_storages[i]).val;

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentStorageNamesTest";
    }

private:
    std::vector<TensorStorageType> _storages{};
    std::vector<std::string>       _expected_vars{};
};

class CLTensorArgumentComponentValuesTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { 12, 14, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentComponentValuesTest()
    {
        _components.push_back(TensorComponentType::Dim0);
        _components.push_back(TensorComponentType::Dim1);
        _components.push_back(TensorComponentType::Dim2);
        _components.push_back(TensorComponentType::Dim3);
        _components.push_back(TensorComponentType::Dim4);
        _components.push_back(TensorComponentType::Dim1xDim2);
        _components.push_back(TensorComponentType::Dim2xDim3);

        _expected_vals.push_back(std::to_string(shape[0]));
        _expected_vals.push_back(std::to_string(shape[1]));
        _expected_vals.push_back(std::to_string(shape[2]));
        _expected_vals.push_back(std::to_string(shape[3]));
        _expected_vals.push_back(std::to_string(shape[4]));
        _expected_vals.push_back(std::to_string(shape[1] * shape[2]));
        _expected_vals.push_back(std::to_string(shape[2] * shape[3]));
    }

    bool run() override
    {
        VALIDATE_ON_MSG(_components.size() == _expected_vals.size(), "The number of components and values does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_tests = _expected_vals.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_tests; ++i)
        {
            CLTensorArgument arg(tensor_name, info, true /* return_dims_by_value */);

            const std::string expected_var_val = _expected_vals[i];
            const std::string actual_var_val   = arg.component(_components[i]).str;

            VALIDATE_TEST(actual_var_val.compare(expected_var_val) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentComponentValuesTest";
    }

private:
    std::vector<TensorComponentType> _components{};
    std::vector<std::string>         _expected_vals{};
};

class CLTensorArgumentComponentsUsedPassByValueFalseTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { 12, 14, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentComponentsUsedPassByValueFalseTest()
    {
        _components.push_back(TensorComponentType::Dim0);
        _components.push_back(TensorComponentType::Dim2);
        _components.push_back(TensorComponentType::Dim3);
        _components.push_back(TensorComponentType::Dim1xDim2);
        _components.push_back(TensorComponentType::OffsetFirstElement);
        _components.push_back(TensorComponentType::Stride1);
        _components.push_back(TensorComponentType::Stride2);
        _components.push_back(TensorComponentType::Stride3);
        _components.push_back(TensorComponentType::Dim0); // Repeat the query. The TensorArgument should not create a new variable
        _components.push_back(TensorComponentType::Dim2); // Repeat the query. The TensorArgument should not create a new variable
        _components.push_back(TensorComponentType::Dim3); // Repeat the query. The TensorArgument should not create a new variable

        _expected_vars.push_back("src_dim0");
        _expected_vars.push_back("src_dim2");
        _expected_vars.push_back("src_dim3");
        _expected_vars.push_back("src_dim1xdim2");
        _expected_vars.push_back("src_offset_first_element");
        _expected_vars.push_back("src_stride1");
        _expected_vars.push_back("src_stride2");
        _expected_vars.push_back("src_stride3");
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_components = _components.size();

        int32_t test_idx = 0;

        CLTensorArgument arg(tensor_name, info, false /* return_dims_by_value */);
        for(size_t i = 0; i < num_components; ++i)
        {
            arg.component(_components[i]);
        }

        const auto actual_vars = arg.components();

        const size_t num_vars = _expected_vars.size();

        VALIDATE_ON_MSG(actual_vars.size() == num_vars, "The number of variables must match the number of expected variables");

        for(size_t i = 0; i < num_vars; ++i)
        {
            // Validate variable name
            const std::string expected_var_name = _expected_vars[i];
            const std::string actual_var_name   = actual_vars[i].str;
            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);

            // Validate data type
            const DataType expected_var_type = DataType::Int32;
            const DataType actual_var_type   = actual_vars[i].desc.dt;
            VALIDATE_TEST(actual_var_type == expected_var_type, all_tests_passed, test_idx++);

            // Validate data type length
            const int32_t expected_var_len = 1;
            const int32_t actual_var_len   = actual_vars[i].desc.len;
            VALIDATE_TEST(actual_var_len == expected_var_len, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentComponentsUsedPassByValueFalseTest";
    }

private:
    std::vector<TensorComponentType> _components{};
    std::vector<std::string>         _expected_vars{};
};

class CLTensorArgumentComponentsUsedPassByValueTrueTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { 12, 14, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentComponentsUsedPassByValueTrueTest()
    {
        _components.push_back(TensorComponentType::Dim0);
        _components.push_back(TensorComponentType::Dim2);
        _components.push_back(TensorComponentType::Dim3);
        _components.push_back(TensorComponentType::Dim1xDim2);
        _components.push_back(TensorComponentType::OffsetFirstElement);
        _components.push_back(TensorComponentType::Stride1);
        _components.push_back(TensorComponentType::Stride2);
        _components.push_back(TensorComponentType::Stride3);
        _components.push_back(TensorComponentType::OffsetFirstElement); // Repeat the query. The TensorArgument should not create a new variable
        _components.push_back(TensorComponentType::Stride1);            // Repeat the query. The TensorArgument should not create a new variable

        _expected_vars.push_back("src_offset_first_element");
        _expected_vars.push_back("src_stride1");
        _expected_vars.push_back("src_stride2");
        _expected_vars.push_back("src_stride3");
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_components = _components.size();

        int32_t test_idx = 0;

        CLTensorArgument arg(tensor_name, info, true /* return_dims_by_value */);
        for(size_t i = 0; i < num_components; ++i)
        {
            arg.component(_components[i]);
        }

        const auto actual_vars = arg.components();

        const size_t num_vars = _expected_vars.size();

        VALIDATE_ON_MSG(actual_vars.size() == num_vars, "The number of variables must match the number of expected variables");

        // Since the dimensions are passed by value, we expect only the variables for the strides
        for(size_t i = 0; i < num_vars; ++i)
        {
            // Validate variable name
            const std::string expected_var_name = _expected_vars[i];
            const std::string actual_var_name   = actual_vars[i].str;
            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);

            // Validate data type
            const DataType expected_var_type = DataType::Int32;
            const DataType actual_var_type   = actual_vars[i].desc.dt;
            VALIDATE_TEST(actual_var_type == expected_var_type, all_tests_passed, test_idx++);

            // Validate data type length
            const int32_t expected_var_len = 1;
            const int32_t actual_var_len   = actual_vars[i].desc.len;
            VALIDATE_TEST(actual_var_len == expected_var_len, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentComponentsUsedPassByValueTrueTest";
    }

private:
    std::vector<TensorComponentType> _components{};
    std::vector<std::string>         _expected_vars{};
};

class CLTensorArgumentStoragesUsedTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { 12, 14, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentStoragesUsedTest()
    {
        _storages.push_back(TensorStorageType::BufferUint8Ptr);
        _storages.push_back(TensorStorageType::Texture2dReadOnly);
        _storages.push_back(TensorStorageType::BufferUint8Ptr); // Repeat the query. The TensorArgument should not create a new variable

        _expected_vars.push_back("src_ptr");
        _expected_vars.push_back("src_img2d");
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_storages = _storages.size();

        int32_t test_idx = 0;

        CLTensorArgument arg(tensor_name, info, true /* return_dims_by_value */);
        for(size_t i = 0; i < num_storages; ++i)
        {
            arg.storage(_storages[i]);
        }

        const auto actual_vars = arg.storages();

        const size_t num_vars = _expected_vars.size();

        VALIDATE_ON_MSG(actual_vars.size() == num_vars, "The number of variables must match the number of expected variables");

        for(size_t i = 0; i < num_vars; ++i)
        {
            // Validate variable name
            const std::string expected_var_name = _expected_vars[i];
            const std::string actual_var_name   = actual_vars[i].val;
            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);

            // Validate storage type
            const std::string expected_var_type = cl_get_variable_storagetype_as_string(_storages[i]);
            const std::string actual_var_type   = actual_vars[i].type;
            VALIDATE_TEST(actual_var_type == expected_var_type, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentStoragesUsedTest";
    }

private:
    std::vector<TensorStorageType> _storages{};
    std::vector<std::string>       _expected_vars{};
};

class CLTensorArgumentComponentsUsedPassByValueTrueDynamicDimTrueTest : public ITest
{
public:
    const DataType    dt          = DataType::Fp32;
    const TensorShape shape       = TensorShape({ { -1, -1, 3, 1, 2 } });
    const std::string tensor_name = "src";

    CLTensorArgumentComponentsUsedPassByValueTrueDynamicDimTrueTest()
    {
        _components.push_back(TensorComponentType::Dim0);
        _components.push_back(TensorComponentType::Dim2);
        _components.push_back(TensorComponentType::Dim3);
        _components.push_back(TensorComponentType::Dim1xDim2);
        _components.push_back(TensorComponentType::OffsetFirstElement);
        _components.push_back(TensorComponentType::Stride1);
        _components.push_back(TensorComponentType::Stride2);
        _components.push_back(TensorComponentType::Stride3);
        _components.push_back(TensorComponentType::OffsetFirstElement); // Repeat the query. The TensorArgument should not create a new variable
        _components.push_back(TensorComponentType::Stride1);            // Repeat the query. The TensorArgument should not create a new variable

        _expected_vars.push_back("src_dim0");
        _expected_vars.push_back("src_dim1xdim2");
        _expected_vars.push_back("src_offset_first_element");
        _expected_vars.push_back("src_stride1");
        _expected_vars.push_back("src_stride2");
        _expected_vars.push_back("src_stride3");
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TensorInfo info(dt, shape, TensorDataLayout::Nhwc, 1);

        const size_t num_components = _components.size();

        int32_t test_idx = 0;

        CLTensorArgument arg(tensor_name, info, true /* return_dims_by_value */);
        for(size_t i = 0; i < num_components; ++i)
        {
            arg.component(_components[i]);
        }

        const auto actual_vars = arg.components();

        const size_t num_vars = _expected_vars.size();

        VALIDATE_ON_MSG(actual_vars.size() == num_vars, "The number of variables must match the number of expected variables");

        // Since the dimensions are passed by value, we expect only the variables for the strides
        for(size_t i = 0; i < num_vars; ++i)
        {
            // Validate variable name
            const std::string expected_var_name = _expected_vars[i];
            const std::string actual_var_name   = actual_vars[i].str;
            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);

            // Validate data type
            const DataType expected_var_type = DataType::Int32;
            const DataType actual_var_type   = actual_vars[i].desc.dt;
            VALIDATE_TEST(actual_var_type == expected_var_type, all_tests_passed, test_idx++);

            // Validate data type length
            const int32_t expected_var_len = 1;
            const int32_t actual_var_len   = actual_vars[i].desc.len;
            VALIDATE_TEST(actual_var_len == expected_var_len, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTensorArgumentComponentsUsedPassByValueTrueDynamicDimTrueTest";
    }

private:
    std::vector<TensorComponentType> _components{};
    std::vector<std::string>         _expected_vars{};
};
} // namespace ckw

#endif // CKW_TESTS_CLTENSORARGUMENTTEST_H
