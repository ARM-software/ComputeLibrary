<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Security Policy

KleidiAI software is verified for security for official releases and as such does not make promises about the quality of
the product for patches delivered between releases.

## Reporting a Vulnerability

Security vulnerabilities may be reported to the Arm Product Security Incident Response Team (PSIRT) by sending an email
to [psirt@arm.com](mailto:psirt@arm.com).

For more information visit https://developer.arm.com/support/arm-security-updates/report-security-vulnerabilities

## Security Guidelines

When KleidiAI is integrated and used in a product, developer must follow the security guidelines to improve security of the product.

- The numerical behaviour of KleidiAI may vary slightly from other micro-kernel implementations,
  between different micro-kernel variants in KleidiAI and between different versions of KleidiAI.
  The user should not be dependent on precise numerical behaviour of KleidiAI.
- KleidiAI micro-kernels do not have limit on the size of the operation it is performing.
  The caller must make sure the size of the operation is suitable for the system
  and does not cause denial-of-service.
- KleidiAI micro-kernels do not perform bound checks on input or output buffers.
  It is the caller’s responsibility to ensure that buffers are correctly sized,
  and the pointer offsets are correctly calculated.

## Third Party Dependencies

Build scripts within this project download third party sources. KleidiAI uses the following third party sources:

- Google Test v1.14.0, for the testing suite.
- Google Benchmark v1.8.4, for the benchmarking suite.
