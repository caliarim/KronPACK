function T = mumpfun(T, Lfun, mu)
% MUMPFUN mu-mode action.
%    S = MUMPFUN(T, Lfun, mu) computes the mu-mode action of the
%    complex tensor T of size m_1 x ... x m_d with the function Lfun
%    (which act on columns of a matrix), that is
%
%    S = T x_{mu} Lfun
%
%    [CCZ23] M. Caliari, F. Cassini, and F. Zivcovich,
%            A mu-mode BLAS approach for multidimensional tensor-structured
%            problems, NUMERICAL ALGORITHMS 92, 2483-2508 (2023)
  if (nargin < 3)
    error('Not enough input arguments.');
  end
  if (isempty(T) || isempty(Lfun) || isempty(mu))
    error('Not enough non-empty input arguments');
  end
  sT = [size(T), ones(1, mu-length(size(T)))];
  if (mu == 1)
    T = Lfun(reshape(T, sT(mu), []));
    sT(mu) = size(T, 1);
    T = reshape(T, sT);
  else
    idx = [mu, 1:(mu-1), (mu+1):length(sT)];
    T = Lfun(reshape(permute(T, idx), sT(mu), []));
    sT(mu) = size(T, 1);
    T = ipermute(reshape(T, sT(idx)), idx);
  end
end
%!test % 1d
%! T = (1:4)';
%! L = reshape (1:8,2,4);
%! Lfun = @(u) L*u;
%! Y = mumpfun(T,Lfun,1);
%! assert(Y,mump(T,L,1))
%!test %2d
%! T = reshape(1:16,4,4);
%! L = reshape (1:8,2,4);
%! Lfun = @(u) L*u;
%! Y1 = mumpfun(T,Lfun,1);
%! assert(Y1,mump(T,L,1))
%! Y2 = mumpfun(T,Lfun,2);
%! assert(Y2,mump(T,L,2))
%!test %2d rect
%! T = randn(2,3);
%! A = randn(4,2);
%! B = randn(5,3);
%! Afun = @(u) A*u;
%! Bfun = @(u) B*u;
%! Y1 = mumpfun(T,Afun,1);
%! assert(Y1,mump(T,A,1))
%! Y2 = mumpfun(T,Bfun,2);
%! assert(Y2,mump(T,B,2))
%!test %2d rect complex
%! T = randn(2,3)+1i*randn(2,3);
%! A = randn(4,2)+1i*randn(4,2);
%! B = randn(5,3)+1i*randn(5,3);
%! Afun = @(u) A*u;
%! Bfun = @(u) B*u;
%! Y1 = mumpfun(T,Afun,1);
%! assert(Y1,mump(T,A,1))
%! Y2 = mumpfun(T,Bfun,2);
%! assert(Y2,mump(T,B,2))
%!test % 3d
%! T = randn(3,4,5);
%! A = randn(4,3);
%! B = randn(5,4);
%! C = randn(6,5);
%! Afun = @(u) A*u;
%! Bfun = @(u) B*u;
%! Cfun = @(u) C*u;
%! Y = mumpfun(T,Afun,1);
%! assert(Y,mump(T,A,1))
%! Y = mumpfun(T,Bfun,2);
%! assert(Y,mump(T,B,2))
%! Y = mumpfun(T,Cfun,3);
%! assert(Y,mump(T,C,3))
%!test % implicit last dimension
%! T = randn(2);
%! A = randn(3,1);
%! Afun = @(u) A*u;
%! Y = mumpfun(T,Afun,3);
%! assert(Y,mump(T,A,3));
%!error
%! mumpfun();
%!error
%! mumpfun(rand(2,3,4))
%!error
%! mumpfun(rand(2,3,4), @(u) rand(2)*u)
%!error
%! mumpfun([],@(u) rand(2)*u,1)
%!error
%! mumpfun(randn(2),[],1)
%!error
%! mumpfun(randn(2),@(u) rand(2)*u,[])
