# Gemm Tuner

## Introduction

This is a set of 2 script tools for tuning the performance of OpenCL GEMM kernels (limited to Convolution layer
functions only for now).  Specifically, we tune 3 GEMM kernels, each has a different implementation **strategy** of the
GEMM operation: **native**, **reshaped**, **reshaped only rhs**. The details of these strategies can be found in the
documentations of the corresponding kernels: **CLGEMMMatrixMultiplyNativeKernel**,
**CLGEMMMatrixMultiplyReshapedKernel** and **CLGEMMMatrixMultiplyReshapedOnlyRHSKernel**.

The outputs of the tuning process are 1 optimal configuration (called **GEMM Configuration** or **GEMMConfig**, for
more details see Approach section) for each of the 3 strategies.

## Location
The 2 scripts **benchmark_gemm_examples.sh** and **GemmTuner.py** can be found under $ACL_ROOT/examples/gemm_tuner.

## Pre-requisite
* A target device to be tuned, plus the following on the device:
    * Android or Linux OS
    * Bash shell
    * Built ACL with benchmark examples binaries
    * benchmark_gemm_examples.sh script
    * gemm shape file

       A csv file containing the **GEMMParam search list**. This is the list of GEMMParams/gemm shapes that we're
       interested in (For more details see Approach section). The default list is prepared by ACL developers in advance
       and can be provided on request.

       The format is described as:

       A headerless csv file with fields separated by commas and commas only (there cannot be whitespaces around each
       field).

       Note also comments and extraneous empty lines are not permitted.

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
      The default lists are prepared by ACL developers in advance and can be provided on request.

      The format of the file for each strategy is the same:  

      A headerless csv file with fields separated by commas and commas only (there cannot be whitespaces around each
      field). Note also comments and extraneous empty lines are not permitted.

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

        A gemm config is a list of 4 positive integers \<m0, n0, k0, h0\> and 2 boolean values interleave_rhs and
        transpose_rhs, with:

        m0 - Number of rows processed by the matrix multiplication  
        n0 - Number of columns processed by the matrix multiplication  
        k0 - Number of partial accumulations performed by the matrix multiplication  
        h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row  
        interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)  
        transpose_rhs - Transpose rhs matrix (1) / Do not transpose rhs matrix (0)  

        Only the following configurations of M0, N0 and K0 are currently supported:

        M0 = 1, 2, 3, 4, 5, 6, 7, 8  
        N0 = 2, 3, 4, 8, 16  
        K0 = 2, 3, 4, 8, 16  
        H0 >= 1  

        An example gemm config file looks like:
  ```
  4,4,4,1,1,1
  4,4,4,3,1,0
  ...
  ```
      * Strategy **reshaped**:

        A gemm config is a list of 5 positive integers \<m0, n0, k0, v0, h0\> and 3 boolean values interleave_lhs,
        interleave_rhs and transpose_rhs, with:

        m0 - Number of rows processed by the matrix multiplication  
        n0 - Number of columns processed by the matrix multiplication  
        k0 - Number of partial accumulations performed by the matrix multiplication  
        v0 - Number of vertical blocks of size (m0xk0) stored on the same output row  
        h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row  
        interleave_lhs - Interleave lhs matrix (1) / Do not interleave lhs matrix (0)  
        interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)  
        transpose_rhs - Transpose rhs matrix but not lhs matrix (1) / Do not transpose rhs matrix but do transpose
        lhs matrix (0)  

        * If rhs matrix is transposed only the following configurations are currently supported:

          M0 = 2, 3, 4, 5, 6, 7, 8  
          N0 = 2, 3, 4, 8, 16  
          K0 = 2, 3, 4, 8, 16  
          V0 >= 1  
          H0 >= 1  

        * If lhs matrix is transposed only the following configurations are currently supported:

          M0 = 2, 3, 4, 8  
          N0 = 2, 3, 4, 8, 16  
          K0 = 2, 3, 4, 8, 16  
          V0 >= 1  
          H0 >= 1  

        An example gemm config file looks like:
  ```
  4,4,4,1,3,1,1,1
  4,4,4,3,3,1,1,0
  ...
  ```
* A host machine, plus these on the machine:
    * python >= 3.6
    * GemmTuner.py script

## Usage
The tuning stage consists of 2 steps:

1. Run benchmarks:

   Run the shell script (**benchmark_gemm_examples.sh**) on your **target device**. Note that all the built benchmark
   examples have to be present on your target device prior to running. The benchmark results will be saved to json
   files in an output directory.
   ```
   Usage: benchmark_gemm_examples.sh [-h] -s \<strategy\> -e \<example_binary_dir\> -g \<gemm_shape_file\>
   -c \<gemm_config_file\> [-o \<out_dir\>]

   Options:
           -h
           Print help messages. If a strategy is specified with -s \<strategy\>, then only display messages relevant
           to that strategy. Otherwise if no strategy is specified, display messages for all available strategies.

           -s \<strategy\>
           Strategy option.
           Options: native reshaped_rhs_only reshaped.

           -e \<example_binary_dir\>
           Path to directory that holds all example binaries

           -g \<gemm_shape_file\>
           Path to gemm shape csv file

           -c \<gemm_config_file\>
           Path to gemm config csv file

           -o \<out_dir\>
           Path to output directory that holds output json files
           Default: out
   ```
2. Run analyser:

  Run the python script (**GemmTuner.py**) on your **host machine**.
  You'll need to transfer all the benchmark result json files generated from the previous step to your host machine
  beforehand. The script will output the best configuration, along with some analysis statistics for each strategy, and
  optionally save the parsed benchmark results into csv files (one for each strategy) for further analysis.
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
                           Path to directory that holds output csv files. One per
                           strategy
     -t TOLERANCE, --tolerance TOLERANCE
                           For testing if two GEMMConfigs are equivalent in terms
                           of performance. The tolerance is OpenCL timer in
                           milliseconds. Recommended value: <= 0.1 ms
     -D, --debug           Enable script debugging output

   ```

## Approach

This section gives a brief description and rationale of the approach adopted by the current version of GEMM Tuner.

As explained in the Introduction section, the outputs of the tuner are 1 optimal GEMMConfig for each strategy.
This is because we can only integrate 1 GEMMConfig for each strategy in ACL at compile time. In theory, however, the
optimal GEMMConfig also depends on different parameters of GEMM (called GEMM Parameter or GEMMParam, e.g.: the shape
of the operation); thus ideally, for each strategy, the optimal configurations should be a mapping from GEMMParam to
GEMMConfig instead of a single GEMMConfig.

To address this issue, we ensure the one single optimal GEMMConfig can generalise well to all potential GEMMParams
(or at least the ones that we care about). The approach we adopt involves a preliminary stage where a collection of
common GEMMParams (GEMM shapes from popular networks) are compiled. Then, to reduce the final tuning time, rather
contradictorily, we spend a lot of time searching for near-optimal GEMMConfigs for each GEMMParam first, and then
discard redundant GEMMParams which share similar optimal GEMMConfigs with others. The resultant list of GEMMParams is
called a __GEMMParam search list__, as in these GEMMParams are typical enough to capture the space of GEMMParams that
we care about.

During this preliminary stage we also produce a list of good GEMMConfigs that can be used to search for the optimal one
in the actual tuning stage. This, again, is to reduce the tuning time, and the resultant list is called a
__GEMMConfig search list__.

The GEMMParam search list and the GEMMConfig search list are investigated and prepared by the developers; the users of
GEMM tuner need not worry about producing them, but they need to obtain them prior to running the tuner.

Once these two lists (2 for each strategy, so 6 in total) are obtained, they can be fed to the tuner, to produce the
optimal GEMMConfig(s).