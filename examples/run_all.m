% Script to produce at once all the results of the experiments
% of [CCZ23, Sec. 4]
%
% List of experiments:
% (1) Code validation
% (2) Hermite-Laguerre-Fourier function decomposition
% (3) Multivariate interpolation
% (4) Linear evolutionary equation
% (5) Semilinear evolutionary equation
%
% [CCZ23] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional tensor-structured
%         problems, NUMERICAL ALGORITHMS 92, 2483-2508 (2023)

clc
clear all
close all

code_validation;
example_spectral;
example_interpolation;
example_exponential;
example_imex;
