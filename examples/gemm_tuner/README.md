# Gemm Tuner

## Pre-requisite
(Preferably) bash shell
Built benchmark examples
python >= 3.6

## Usage
The tuning consists of 2 steps:

1. Run benchmarks: Run the runner shell script (benchmark_gemm_examples.sh) on
your target device. Note that all the built benchmark examples have to be
present on your target device prior to running. The script will run the selected
strategy, over all pre-defined tunable configurations, on a set of gemm shapes
provided by the user, and then save the benchmark results to json files in an
output directory.

[$SHELL] ./benchmark_gemm_examples.sh -s \<strategy\> -e \<example_binary_dir\>
    -g \<gemm_shape_file\> -c \<gemm_config_file\> [-o \<out_dir\>]

2. Run analyser: Run the python script (GemmTuner.py) on your host device.
You'll need to transfer all the benchmark result json files generated from the
previous step to your host machine beforehand. Note that this requires
python >= 3.6. The script will output the best configuration, along with some
analysis statistics for each strategy, and optionally save the parsed benchmark
results into csv files (one for each strategy) for further analysis.

python GemmTuner.py -b \<benchmark_results_dir\> [-t \<tolerance\>]
[-o \<out_dir\>]
