%  Scripts to reproduce the numerical experiments of [CCZ23]
%
%  [CCZ23] M. Caliari, F. Cassini, and F. Zivcovich,
%          A mu-mode BLAS approach for multidimensional tensor-structured
%          problems, NUMERICAL ALGORITHMS 92, 2483-2508 (2023)
%
%  Numerical experiments
%    code_validation*      - Code validation
%    example_spectral      - Hermite-Laguerre-Fourier function decomposition
%    example_interpolation - Multivariate interpolation
%    example_exponential   - Linear evolutionary equation
%    example_imex          - Semilinear evolutionary equation
%    run_all               - Run all the numerical experiments
%    lagpts^               - Gauss-Laguerre quadrature points and weights
%    gammaratio^           - Ratio of gamma functions
%    hermpts^              - Gauss-Hermite quadrature points and weights
%
%  * requires Tensorlab and Tensor Toolbox for MATLAB, hence it can't
%    be executed in GNU Octave.
%
%  ^ taken from Chebfun repository https://github.com/chebfun/chebfun
