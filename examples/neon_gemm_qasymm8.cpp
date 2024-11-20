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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include <NEArithmeticAddition.h>


using namespace arm_compute;
int main(){
        std::vector<float> input_1(1*64*32*100,-1);
        std::vector<float> input_2(1*64*32*100,1);
        std::vector<float> output(1*64*32*100,1);

        TensorInfo input_info_1 = TensorInfo(TensorShape(1,64,32,100),1,DataType::F32,DataLayout::NCH>        TensorInfo input_info_2 = TensorInfo(TensorShape(1,64,32,100),1,DataType::F32,DataLayout::NCH>        TensorInfo output_info = TensorInfo(TensorShape(1,64,32,100),1,DataType::F32,DataLayout::NCHW>

        Status status = NEArithmeticAddition::validate(&input_info1,&input_info_2, &output_info, Conv>        std::cout << status.error_description() << std::endl;

        Tensor input_tensor_1,input_tensor_2, output_tensor;
        input_tensor_1.allocator()->init(input_info_1);
        input_tensor_1.allocator()->import_memory(input_1.data());

        input_tensor_2.allocator()->init(input_info_2);
        input_tensor_2.allocator()->import_memory(input_2.data());

        output_tensor.allocator()->init(output_info);
        output_tensor.allocator()->import_memory(output.data());

        NEArithmeticAddition neArAdd;
        neArAdd.configure(&input_tensor_1,&input_tensor_2,&output_info, ConvertPolicy::SATURATE, Acti>

        std::cout << "output before run: " << output[52] << std::endl;
        neArAdd.run();
        std::cout << "output after run: " << output[52] << std::endl;

        return 0;
}



