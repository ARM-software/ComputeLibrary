# Extract list of operators from a list of TfLite models

## Introduction

Purpose of this script is to inspect a list of user-provided TfLite models and report
the list of operators that are used as well as the data-types that the models operate on.
The script can subsequently generate a configuration file that can be provided to the
Compute Library build system and generate a library that contains only the operators required
by the given model(s) to run.

Utilizing this script, use-case tailored Compute Library dynamic libraries can be created,
helping reduce the overall binary size requirements.

## Usage example

Assuming that the virtual environment is activated and the requirements are present,
we can run the following command:

```bash
./report_model_ops.py -m modelA.tfile modelB.tflite -c build_config.json
```

## Input arguments

***models (required)*** :
A list of comma separated model files.

Supported model formats are:

* TfLite

***config (optional)*** :
The configuration file to be created on JSON format that can be provided to ComputeLibrary's
build system and generate a library with the given list of operators and data-types

***debug (optional)*** :
Flag that enables debug information
