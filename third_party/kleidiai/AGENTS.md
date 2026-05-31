<!--
    SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# KleidiAI Repo Cheatsheet

## Purpose

- KleidiAI delivers optimized Arm® micro-kernels (packing + matmul) for
  AI/ML frameworks that already manage scheduling, threading, and memory.

## Key Layout

- `kai/` – Common headers plus micro-kernel families grouped by operator (for
  example `ukernels/matmul`). Each family follows naming rules documented in
  per-directory READMEs and implements the standard interfaces in
  `ukernels/matmul/*.h`.
- `test/` – GoogleTest unit suites covering micro-kernel correctness and API
  guarantees.
- `benchmark/` – CMake targets that time kernels across Arm variants.
- `examples/` – Minimal builds that exercise the library as an external
  dependency and serve as smoke tests.
- `docs/` – Task-focused guides (packing/matmul intros, indirect matmul
  walkthroughs, framework integration examples, external patches).
- `docker/` - Contains containers used in CI.

## Build & Run

KleidiAI uses two build systems; CMake and Bazel.

- Native Arm build (default): `cmake -S . -B build && cmake --build build`
  - Test with `build/kleidiai_test`
- Bazel build and test `bazelisk test //test:kleidiai_test`

### CI/CD

The testing pipeline is described in `.gitlab-ci.yml`, which does make use of
container described in `docker/Dockerfile`. This utilizes FVP, which enables
testing on different HW configurations.

## Working Notes for Agents

- Micro-kernel variants are contained in `.S`, `.c` and `.h` files, named
  `kai_<op>_<variant>_<tech>_<feature>`, and only depend on `kai_common.h`.
- Typical flow: pack LHS → pack RHS → invoke matmul kernel.

## Review Checklist

Use items here as checklist when reviewing, or making changes,

### General

- Remove commented-out code before committing.

- Add new source files to all relevant build scripts (CMake and Bazel).

- Update copyright notices to include the current year. For Markdown and image
  files, the notice lives in a `.license` sidecar.

- Use `TODO` comments sparingly and explain the follow-up.

- Write clear, correctly formatted documentation for every function:

  ```
  /// Brief one-line description.
  ///
  /// Optional longer description if needed
  ///
  /// @param[in/out] param_name Description of parameter
  ///
  /// @return Description of return value, if any.
  ```

- Use the term _micro-kernel_ rather than kernel/ukernel/function.

- Guard CPU-feature-dependent code with
  [ACLE feature test macros][acle_features] (noting that `__ARM_FEATURE_SME` is
  often an exception).

### Commit Message

- Describe the change; one-line messages are reserved for trivial edits.
- Include `Signed-off-by: <Contact>` in every commit.

### CHANGELOG

- Document all micro-kernel additions, optimizations, and bug fixes in
  `CHANGELOG.md`

### New Micro-kernels

- Micro-kernel bundles (`.c`, `.h`, `.S`) live under `kai/`; pack kernels go in
  `pack`, matmul kernels in operator-specific directories with interface headers
  like `kai_imatmul_clamp_f16_f16p_f16p_interface.h`.
- Follow the established naming convention or explicitly extend it.
- Provide matching interface headers for matmul micro-kernels.
- Add unit tests for every new kernel; an end-to-end pack + matmul test is
  acceptable if per-kernel tests are impractical.
- Register new matmul-style kernels with the benchmark suite.

#### Pure Assembly Micro-kernels

- Conform to [AAPCS64]; save `d8`–`d15` and `x19`–`x28` if modified.
- Emit exactly one `ret` and avoid calling other functions (`bl`), except
  approved [SME support routines][sme_support] with proper `LR`/`FP`
  preservation and `__ARM_FEATURE_SME` guards.
- Ensure the implementation matches the advertised behavior (e.g., clamp
  functionality).

### Updating Micro-kernels

- Preserve public symbol names unless the change and changelog clearly state
  otherwise; parameter renames for consistency are fine.
- Maintain kernel functionality unless explicitly changing it.

### CMakeLists and bazel

- Keep file listings sorted.
- File lists in `BUILD.bazel` are named `<TECH>[_<FEAT>]*_KERNELS[_ASM]`.
- File lists in `CMakeLists.txt` are named
  `KLEIDIAI_FILES_<TECH>[_<FEAT>]*[_ASM]`.
- Kernels should be added to list that matches required technology and features.
  - Kernels that use inline assembly should be added to a list without `_ASM`
    suffix. However, this prevent some compiler support, so avoid if possible.
  - Kernels that doesn't use inline assembly should be added to
    `KLEIDIAI_FILES_<TYPE>_ASM`. This is typically what we want to do.

### Tests

- Never skip tests unless unavoidable.
- Cache expensive reference data generation (notably for matmul) using utilities
  like `common/cache.hpp`.
- Test code must use `KAI_ASSUME_ALWAYS(expr)` and `KAI_ASSERT_ALWAYS(expr)` as
  to not be optimized away in release builds.

### Common Problems

- Confirm `KAI_UNUSED(symbol)` values are actually unused.

- Enforce required parameter values via `KAI_ASSUME(param == value)`.

- Use `KAI_ASSERT(<assumption>)` for other invariants.

- At the top of `.c` files, assert required architecture extensions, for
  example:

  ```c
  #if (!defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)) && \
      !defined(_M_ARM64)
  ```

[aapcs64]: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
[acle_features]: https://arm-software.github.io/acle/main/acle.html#feature-test-macros
[sme_support]: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#sme-support-routines
