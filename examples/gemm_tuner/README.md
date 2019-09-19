# Gemm Tuner

## Pre-requisite
(Preferably) bash shell
benchmark examples

## Usage
Run gemm examples of a selected strategy, over all pre-defined tunable configurations, on a set of gemm shapes provided
by the user. Save the benchmark results to json files in an output directory.

[$SHELL] ./benchmark_gemm_examples.sh -e \<example_binary_dir\> -s \<strategy\> -g \<gemm_shape_file\> -c \<gemm_config_file\> [-o \<out_dir\>, [-i \<iteration\>]]
