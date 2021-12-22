function T = mumat(T, mu)
% MUMAT mu-matricization.
%     S = MUMAT(T, mu) computes the mu-matricization of the complex tensor T
%     of size m_1 x ... x m_d, that is
%
%     S = T^{(mu)}
%
%     where S is the matrix of size m_{mu} x (m_1...m_{mu-1}m_{mu+1}...m_d)
%     having the mu-fibers of T as columns.
%
%    [CCZ21] M. Caliari, F. Cassini, and F. Zivcovich,
%            A mu-mode BLAS approach for multidimensional tensor-structured
%            problems, Submitted 2021
  if (nargin < 2)
    error('Not enough input arguments.');
  end
  if (isempty(T) || isempty(mu))
    error('Not enough non-empty input arguments');
  end
  mmu = size(T, mu);
  if (mu == 1)
    T = reshape(T, mmu, []);
  else
    T = reshape(permute(T, [mu, 1:(mu-1), (mu+1):length(size(T))]), mmu, []);
  end
end
%!test % 1d
%! T = (1:4)';
%! Y = mumat(T,1);
%! assert(Y,T)
%!test %2d
%! T = reshape(1:16,4,4);
%! Y1 = mumat(T,1);
%! assert(Y1,T)
%! Y2 = mumat(T,2);
%! assert(Y2,T.')
%!test %2d rect
%! T = randn(2,3);
%! Y1 = mumat(T,1);
%! assert(Y1,T)
%! Y2 = mumat(T,2);
%! assert(Y2,T.')
%!test %2d rect complex
%! T = randn(2,3)+1i*randn(2,3);
%! Y1 = mumat(T,1);
%! assert(Y1,T)
%! Y2 = mumat(T,2);
%! assert(Y2,T.')
%!test % 3d
%! T = reshape(1:24,2,3,4);
%! Y1 = mumat(T,1);
%! ref = reshape(1:24,2,12);
%! assert(Y1,ref)
%! Y2 = mumat(T,2);
%! ref = [1,2,7,8,13,14,19,20;
%!        3,4,9,10,15,16,21,22;
%!        5,6,11,12,17,18,23,24];
%! assert(Y2,ref)
%! Y3 = mumat(T,3);
%! ref = [1,2,3,4,5,6;
%!        7,8,9,10,11,12;
%!        13,14,15,16,17,18;
%!        19,20,21,22,23,24];
%! assert(Y3,ref)
%!error
%! mumat();
%!error
%! mumat(rand(2,3,4))
%!error
%! mumat([],1)
%!error
%! mumat(randn(2),[])
