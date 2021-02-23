# Gemm Tuner

## Introduction

This is a set of tools for tuning the performance of OpenCL GEMM kernels.  Specifically, we tune 3 GEMM kernels, each
has a different implementation **strategy** of the GEMM operation: **native**, **reshaped**, **reshaped only rhs**.
The details of these strategies can be found in the documentations of the corresponding kernels:
**CLGEMMMatrixMultiplyNativeKernel**, **CLGEMMMatrixMultiplyReshapedKernel** and
**CLGEMMMatrixMultiplyReshapedOnlyRHSKernel**.

The Tuner consists of 2 scripts and 3 binaries:
* cl_gemm_benchmark and GemmTuner.py under examples/gemm_tuner, and
* benchmark_cl_gemm_native, benchmark_cl_gemm_reshaped_rhs_only and benchmark_cl_gemm_reshaped under
  build/tests/gemm_tuner (you'll need to build the library first)

The inputs to the Tuner are a list of 4 valued tuples we call **GEMM shape** or **GEMMParam** (M, N, K, B, and possibly
data type). They define the "shape" and other parameters (eg. data type) of a GEMM operation:
```
LHS x RHS = DST
```
Where LHS is of shape MxK, RHS is of shape KxN and DST is of shape MxN, and B is the batch size.

The outputs of the tuning process are 4 json files:
1. gemm_type_selection.json: selects which kernel type is the best for each GEMMParam
2. gemm_config_native.json: selects a list of best **GEMMConfigs** of the native kernel for each GEMMParam
3. gemm_config_reshapedonlyrhs.json: selects a list of best GEMMConfigs of the reshaped_only_rhs kernel for each GEMMParam
4. gemm_config_reshaped.json: selects a list of best GEMMConfigs of the reshaped kernel for each GEMMParam

These 4 files are the current representations we use for what we call the **heuristics** of a GEMM op: given a GEMMParam,
what kernel and subsequently what configurations for that kernels are the most performant.

## Step-by-step example

### Step1: Prepare the shape and configs files
1. We first need to identify the shapes that we are interested in and store them in a csv file, say *gemm_shapes.csv*.
2. Then we need to specify a set of good GEMMConfig candidates for each kernel in 3 separate csv files (this requires
    some prior heuristics, but can be provided by the Compute Library developers upon requests, based on your target device).

   Say we have *gemm_configs_native.csv", "gemm_configs_reshaped.csv" and "gemm_configs_reshaped_only_rhs.csv".

   Please refer to the Prerequisite section for more details

### Step2: Push relevant files to the target device
All the files that need to be present on the target device are:
* benchmark script: \<ComputeLibrary\>/examples/gemm_tuner/cl_gemm_benchmark
* shapes and configs csv files: gemm_shapes.csv, gemm_configs_native.csv, gemm_configs_reshaped_only_rhs.csv, gemm_configs_reshaped.csv
* Example benchmark binaries: \<ComputeLibrary\>/build/tests/gemm_tuner/benchmark_cl_gemm*

### Step3: Collect benchmark data
With these files on device, we can collect benchmark data using the script. Assume all the example binaries are pushed
to a folder called *gemm_tuner*. While logged onto our device:
```
# Native
./cl_gemm_benchmark -s native -e ./gemm_tuner -g ./gemm_shapes.csv -c ./gemm_configs_native.csv -o results/native
# Reshaped Only RHS
./cl_gemm_benchmark -s reshaped_rhs_only -e ./gemm_tuner -g ./gemm_shapes.csv -c ./gemm_configs_reshaped_only_rhs.csv -o results/reshaped_only_rhs
# Reshaped
./cl_gemm_benchmark -s reshaped -e ./gemm_tuner -g ./gemm_shapes.csv -c ./gemm_configs_reshaped.csv -o results/reshaped
```
You can repeat the 3 commands above to have a bit redundancy in your benchmark data (as you can imagine, measurement is noisy),
but you may need to change the output folder for each repeat

It is also possible to split the benchmark phase among different platforms using the **-i** and **-n** options to specificy the starting experiment and the number of benchmark to run.

# Reshaped benchmark on 3 different platforms
## Platform 1
./cl_gemm_benchmark -s reshaped -e ./gemm_tuner -g ./gemm_shapes.csv -c ./gemm_configs_reshaped.csv -o results/reshaped -i 0 -n 8
## Platform 2
./cl_gemm_benchmark -s reshaped -e ./gemm_tuner -g ./gemm_shapes.csv -c ./gemm_configs_reshaped.csv -o results/reshaped -i 8 -n 8
## Platform 3
./cl_gemm_benchmark -s reshaped -e ./gemm_tuner -g ./gemm_shapes.csv -c ./gemm_configs_reshaped.csv -o results/reshaped -i 16 -n 8

### Step4: Generate the heuristics
1. After benchmarking, we pull the benchmark data, the *results* folder, from the target device to our host machine
2. We use the GemmTuner.py script to give us the heuristics
   ```
   python3 <ComputeLibrary>/examples/gemm_tuner/GemmTuner.py -b ./results -o heuristics
   ```
   When it's finished, there should be 4 json files in the *heuristics* folder

One thing to notice is that the config heuristics might give more than 1 recommendations for each GEMMParam, because
we accept all good GEMMConfigs with a tolerance. If you want fewer recommendations, you can decrease the tolerance by
passing a lower value to *-t \<tolerance\>* to the GemmTuner.py script.

## Prerequisite
* A target device to be tuned, plus the following on the device:
    * Android or Linux OS
    * Bash shell
    * Built Compute Library with benchmark examples binaries
    * cl_gemm_benchmark script
    * gemm shape file

       A csv file containing the **GEMMParam search list**. This is the list of GEMMParams/gemm shapes that we're
       interested in (For more details see Approach section). The default list is prepared by Compute Library developers in advance
       and can be provided on request.

       The format is described as:

       A headerless csv file with fields separated by commas.

       A gemm shape is a list of 4 positive integers \<M, N, K, B\> describing the shapes of the two matrices (LHS and
       RHS) with:

       M - Number of lhs matrix rows  
       N - Number of rhs matrix columns  
       K - Number of lhs matrix columns/rhs matrix rows  
       B - Batch size  

       An example gemm shape file looks like:
  ```
  100,100,30,1
  100,100,30,3
  ...
  ```
    * gemm config file  
      A csv file containing the **GEMMConfig search list**. This is the list of candidate GEMMConfigs among which we
      search for the optimal one. **Note that we have a different list for each strategy.**
      The default lists are prepared by Compute Library developers in advance and can be provided on request.

      The format of the file for each strategy is the same:  

      A headerless csv file with fields separated by commas.

      However the fields of GEMMConfig differ for each strategy:

      * Strategy **native**:
        A gemm config is a list of 3 positive integers \<m0, n0, k0\>, with:

        m0 - Number of rows processed by the matrix multiplication  
        n0 - Number of columns processed by the matrix multiplication  
        k0 - Number of partial accumulations performed by the matrix multiplication

        Only the following configurations of M0, N0 and K0 are currently supported:

        M0 = 1, 2, 3, 4, 5, 6, 7, 8  
        N0 = 2, 3, 4, 8, 16  
        K0 = 2, 3, 4, 8, 16  

        An example gemm config file looks like:
  ```
  1,4,4
  2,3,8
  ...
  ```
      * Strategy **reshaped_rhs_only**:
        A gemm config is a list of 4 positive integers <m0, n0, k0, h0> and 3 boolean values:

        m0 - Number of rows processed by the matrix multiplication  
        n0 - Number of columns processed by the matrix multiplication  
        k0 - Number of partial accumulations performed by the matrix multiplication  
        h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row  
        interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)  
        transpose_rhs - Transpose rhs matrix (1) / Do not transpose rhs matrix (0)  
        export_to_cl_image_rhs - Export rhs matrix to cl_image (1) / Do not export rhs matrix to cl_image (0). Can only be true
                                with certain combinations of the GEMMParams and other configs. Please refer to CLGEMMReshapeRHSMatrixKernel
                                for more details

        Only the following configurations of M0, N0 and K0 are currently supported:

        M0 = 1, 2, 3, 4, 5, 6, 7, 8  
        N0 = 2, 3, 4, 8, 16  
        K0 = 2, 3, 4, 8, 16  
        H0 >= 1  

        An example gemm config file looks like:
  ```
  4,4,4,1,1,1,0
  4,4,4,3,1,0,1
  ...
  ```
      * Strategy **reshaped**:
        A gemm config is a list of 5 positive integers <m0, n0, k0, v0, h0> and 4 boolean values:

        m0 - Number of rows processed by the matrix multiplication  
        n0 - Number of columns processed by the matrix multiplication  
        k0 - Number of partial accumulations performed by the matrix multiplication  
        v0 - Number of vertical blocks of size (m0xk0) stored on the same output row  
        h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row  
        interleave_lhs - Interleave lhs matrix (1) / Do not interleave lhs matrix (0)  
        interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)  
        transpose_rhs - Transpose rhs matrix but not lhs matrix (1) / Do not transpose rhs matrix but do transpose lhs matrix (0)  
        export_to_cl_image_rhs - Export rhs matrix to cl_image (1) / Do not export rhs matrix to cl_image (0). Can only be true
                                with certain combinations of the GEMMParams and other configs. Please refer to CLGEMMReshapeRHSMatrixKernel
                                for more details

        If rhs matrix is transposed only the following configurations are currently supported:

        M0 = 2, 3, 4, 5, 6, 7, 8  
        N0 = 2, 3, 4, 8, 16  
        K0 = 2, 3, 4, 8, 16  
        V0 >= 1  
        H0 >= 1  

        If lhs matrix is transposed only the following configurations are currently supported:

        M0 = 2, 3, 4, 8  
        N0 = 2, 3, 4, 8, 16  
        K0 = 2, 3, 4, 8, 16  
        V0 >= 1  
        H0 >= 1  

        An example gemm config file looks like:
  ```
  4,4,4,1,3,1,1,1,0
  4,4,4,3,3,1,1,0,1
  ...
  ```
* A host machine, plus these on the machine:
    * python >= 3.6
    * GemmTuner.py script

## Usage
The usage of the 2 scripts:

1. cl_gemm_benchmark

   Run the shell script (**cl_gemm_benchmark**) on your **target device**. Note that all the built benchmark
   examples: build/tests/gemm_tuner/benchmark_cl_gemm*, have to be present on your target device prior to running.
   The benchmark results will be saved to json files in an output directory.
   ```
   Usage: cl_gemm_benchmark [-h] -s \<strategy\> -e \<example_binary_dir\> -g \<gemm_shape_file\>
   -c \<gemm_config_file\> [-d \<data_type\>] [-o \<out_dir\>]

   Options:
           -h
           Print help messages. If a strategy is specified with -s <strategy>, then only display messages relevant to that
           strategy. Otherwise if no strategy is specified, display messages for all available strategies.

           -s <strategy>
           Strategy option.
           Options: ${ALL_STRATEGY_OPTIONS[@]}.

           -e <example_binary_dir>
           Path to directory that holds all example binaries

           -g <gemm_shape_file>
           Path to gemm shape csv file

           -c <gemm_config_file>
           Path to gemm config csv file

           -d <data_type>
           Data type option with which to run benchmark examples
           Default: ${DEFAULT_DATA_TYPE}
           Supported options:
           Strategy            :    Data Types
           Native              :    F32
           Reshaped            :    F16, F32
           Reshaped RHS Only   :    F16, F32

           -o <out_dir>
           Path to output directory that holds output json files
           Default: ${DEFAULT_OUT_DIR}
   ```
2. GemmTuner.py:

  Run the python script (**GemmTuner.py**) on your **host machine**.
  You'll need to transfer all the benchmark result json files generated from the previous step to your host machine
  beforehand. The script will output the best kernel and gemm configurations for each gemm param in the 4 output json files
   ```
   Usage: GemmTuner.py [-h] -b PATH [-o PATH] [-t TOLERANCE] [-D]

   CL GEMM Tuner
   optional arguments:
     -h, --help            show this help message and exit
     -b PATH, --benchmark_results PATH
                           Path to benchmark result directory, where benchmark
                           result json files have a file extension of
                           'gemmtuner_benchmark'
     -o PATH, --output_dir PATH
                           Path to directory that holds output json files.
     -t TOLERANCE, --tolerance TOLERANCE
                           For testing if two GEMMConfigs are equivalent in terms
                           of performance. The tolerance is OpenCL timer in
                           milliseconds. Recommended value: <= 0.1 ms
     -D, --debug           Enable script debugging output

   ```
