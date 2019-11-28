# Gemm Tuner

## Introduction

This is a set of 2 script tools for tuning the performance of OpenCL GEMM
kernels (limited to Convolution layer functions only for now).  Specifically, we
tune 3 GEMM kernels, each has a different implementation strategy of the GEMM
operation: native, reshaped, reshaped only rhs. The details of these strategies
can be found in the documentations of the corresponding kernels:
CLGEMMMatrixMultiplyNativeKernel, CLGEMMMatrixMultiplyReshapedKernel and
CLGEMMMatrixMultiplyReshapedOnlyRHSKernel.

The outputs of the tuning process are 1 optimal configuration (called GEMM
Configuration or GEMMConfig) for each of the 3 strategies.

## Approach

This section gives a brief description and rationale of the approach adopted by
the current version of GEMM Tuner.

As explained in the Introduction section, the outputs of the tuner are 1 optimal
GEMMConfig for each strategy. This is because we can only integrate 1 GEMMConfig
for each strategy in ACL at compile time. In theory, however, the optimal
GEMMConfig also depends on different parameters of GEMM (called GEMM Parameter
or GEMMParam, e.g.: the shape of the operation); thus ideally, for each
strategy, the optimal configurations should be a mapping from GEMMParam to
GEMMConfig instead of a single GEMMConfig.

To address this issue, we ensure the one single optimal GEMMConfig can
generalise well to all potential GEMMParams (or at least the ones that we care
about). The approach we adopt involves a preliminary stage where a collection of
common GEMMParams (GEMM shapes from popular networks) are compiled. Then, to
reduce the final tuning time, rather contradictorily, we spend a lot of time
searching for near-optimal GEMMConfigs for each GEMMParam first, and then
discard redundant GEMMParams which share similar optimal GEMMConfigs with
others. The resultant list of GEMMParams is called a __GEMMParam archetype
list__, as in these GEMMParams are typical enough to capture the space of
GEMMParams that we care about.

During this preliminary stage we also produce a list of good GEMMConfigs that
can be used to search for the optimal one in the actual tuning stage. This,
again, is to reduce the tuning time, and the resultant list is called a
__GEMMConfig search list__.

The GEMMParam archetype list and the GEMMConfig search list are investigated and
prepared by the developers; the users of GEMM tuner need not worry about
producing them, but they need to obtain them prior to running the tuner.

Once these two lists (2 for each strategy, so 6 in total) are obtained, they can
be fed to the tuner, to produce the optimal GEMMConfig(s).

## Pre-requisite
* A target device (Android phones, Linux boards, e.t.c.), on which to tune the
  GEMM kernels, plus these on the device:
    * (Preferably) Bash shell
    * Built ACL with benchmark examples
    * GEMMParam archetype list
    * GEMMConfig search list
* A host machine, plus these on the machine:
    * python >= 3.6

## Usage

The tuning stage consists of 2 steps:

1. Run benchmarks: Run the runner shell script (benchmark_gemm_examples.sh) on
your target device. Note that all the built benchmark examples have to be
present on your target device prior to running. The script will run the selected
strategy, over all configs defined in GEMMConfig search list, on all GEMMParams
inside the GEMMParam archetype list, and then save the benchmark results to json
files in an output directory.
```
[$SHELL] ./benchmark_gemm_examples.sh -s \<strategy\> -e \<example_binary_dir\>
-g \<gemmparam_archetype_list\> -c \<gemmconfig_search_list\> [-o \<out_dir\>]
```
2. Run analyser: Run the python script (GemmTuner.py) on your host machine.
You'll need to transfer all the benchmark result json files generated from the
previous step to your host machine beforehand. Note that this requires python >=
3.6. The script will output the best configuration, along with some analysis
statistics for each strategy, and optionally save the parsed benchmark results
into csv files (one for each strategy) for further analysis.
An optional tolerance in milliseconds in OpenCl timer is provided to determine
how far apart in performance two GEMMConfigs have to be, to be considered
different. A default value of 0.01 ms is used, and it's recommended this value
should be < 0.1 ms.
```
python GemmTuner.py -b \<benchmark_results_dir\> [-t \<tolerance\>]
[-o \<out_dir\>]
```
