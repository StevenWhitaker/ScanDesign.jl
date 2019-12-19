# ScanDesign.jl
This package implements an MRI scan parameter optimization technique
presented in
[G. Nataraj, J.-F. Nielsen, and J. A. Fessler. Optimizing MR Scan Design
for Model-Based T1, T2 Estimation from Steady-State Sequences.
IEEE Trans. Med. Imag., 36(2):467-77, February 2017]
(https://ieeexplore.ieee.org/document/7582547).
This code was inspired by the MATLAB code written by Gopal Nataraj,
which can be found [here](https://github.com/gopal-nataraj/scn-dsgn).

## Getting Started
At the Julia REPL, type `]` to enter the package prompt.
Then type `add https://github.com/StevenWhitaker/ScanDesign.jl#v0.0.1`
to add ScanDesign v0.0.1
(note that `v0.0.1` can be replaced with whatever version is needed).
Hit backspace to return to the normal Julia prompt,
and then type `using ScanDesign` to load the package.

## Overview
The function `scandesign` provides the main functionality.
It takes as input an initial set of scan parameters
and a cost function to minimize
(as well as optional parameters for fine-grained control of the optimization).
The function `expectedcost` is the cost function used
in the paper mentioned above that takes an expectation
over a range of unknown and known parameters
of the trace of the inverse Fisher information matrix.
The function `fisher` computes the Fisher information matrix
as described in the aforementioned paper.
