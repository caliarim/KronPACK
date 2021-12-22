% Script to produce in one shot all the results of the examples
% of [CCZ21, section 4]
%
% List of examples:
% (1) Hermite-Laguerre-Fourier function approximation
% (2) Multivariate interpolation
% (3) Linear evolutionary equation
% (4) Semilinear evolutionary equation
%
% [CCZ21] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional
%         tensor-structured problems, Submitted 2021

clc
clear all
close all

example_spectral;
example_interpolation;
example_exponential;
example_imex;
