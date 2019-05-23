/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

class NESGEMMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        alpha = 1.0f;
        beta  = 0.0f;

        std::ifstream stream;
        if(argc > 1)
        {
            stream.open(argv[1], std::fstream::in);
        }

        if(argc < 3 || (argc < 4 && stream.bad()))
        {
            // Print help
            std::cout << "Usage: 1) ./build/neon_sgemm input_matrix_1.npy input_matrix_2.npy [input_matrix_3.npy] [alpha = 1] [beta = 0]\n";
            std::cout << "       2) ./build/neon_sgemm M N K [alpha = 1.0f] [beta = 0.0f]\n\n";
            std::cout << "Too few or no input_matrices provided. Using M=7, N=3, K=5, alpha=1.0f and beta=0.0f\n\n";

            src0.allocator()->init(TensorInfo(TensorShape(5U, 7U), 1, DataType::F32));
            src1.allocator()->init(TensorInfo(TensorShape(3U, 5U), 1, DataType::F32));
            src2.allocator()->init(TensorInfo(TensorShape(3U, 7U), 1, DataType::F32));
        }
        else
        {
            if(stream.good()) /* case file1.npy file2.npy [file3.npy] [alpha = 1.0f] [beta = 0.0f] */
            {
                npy0.open(argv[1]);
                npy0.init_tensor(src0, DataType::F32);
                npy1.open(argv[2]);
                npy1.init_tensor(src1, DataType::F32);

                if(argc > 3)
                {
                    stream.close();
                    stream.clear();
                    stream.open(argv[3], std::fstream::in);
                    if(stream.good()) /* case with third file */
                    {
                        npy2.open(argv[3]);
                        npy2.init_tensor(src2, DataType::F32);

                        if(argc > 4)
                        {
                            // Convert string to float
                            alpha = strtof(argv[4], nullptr);

                            if(argc > 5)
                            {
                                // Convert string to float
                                beta = strtof(argv[5], nullptr);
                            }
                        }
                    }
                    else /* case without third file */
                    {
                        alpha = strtof(argv[3], nullptr);

                        if(argc > 4)
                        {
                            beta = strtof(argv[4], nullptr);
                        }
                    }
                }
            }
            else /* case M N K [alpha = 1.0f] [beta = 0.0f] */
            {
                size_t M = strtol(argv[1], nullptr, 10);
                size_t N = strtol(argv[2], nullptr, 10);
                size_t K = strtol(argv[3], nullptr, 10);

                src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
                src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
                src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

                if(argc > 4)
                {
                    alpha = strtof(argv[4], nullptr);

                    if(argc > 5)
                    {
                        beta = strtof(argv[5], nullptr);
                    }
                }
            }
        }

        init_sgemm_output(dst, src0, src1, DataType::F32);

        // Configure function
        sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta);

        // Allocate all the images
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill the input images with either the data provided or random data
        if(npy0.is_open())
        {
            npy0.fill_tensor(src0);
            npy1.fill_tensor(src1);

            output_filename = "sgemm_out.npy";
            is_fortran      = npy0.is_fortran();

            if(npy2.is_open())
            {
                src2.allocator()->allocate();
                npy2.fill_tensor(src2);
            }
        }
        else
        {
            src2.allocator()->allocate();

            fill_random_tensor(src0, -1.f, 1.f);
            fill_random_tensor(src1, -1.f, 1.f);
            fill_random_tensor(src2, -1.f, 1.f);
        }

        // Dummy run for CLTuner
        sgemm.run();

        return true;
    }
    void do_run() override
    {
        // Execute the function
        sgemm.run();
    }
    void do_teardown() override
    {
        if(!output_filename.empty()) /* Save to .npy file */
        {
            save_to_npy(dst, output_filename, is_fortran);
        }
    }

private:
    Tensor      src0{}, src1{}, src2{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    bool        is_fortran{};
    std::string output_filename{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NESGEMMExample>(argc, argv);
}
