function kv = kronsumv(T, varargin)
% KRONSUMV Action of a Kronecker sum on a tensor.
%    KV = KRONSUMV(T, L) computes the sum
%
%       KV = T x_1 L{1} + T x_2 L{2} + ... + T x_d L{d}
%
%    where T is a complex tensor of size m_1 x ... x m_d, L a cell array of
%    complex matrices (L{mu} of size m_{mu} x m_{mu}) and x_mu denotes the
%    mu-mode product. For d > 1, the vector KV(:) corresponds to the vector
%
%    KRONSUM(L)*T(:)
%
%    KV = KRONSUMV(T, L1, L2, ..., Ld) computes the sum
%
%       KV = T x_1 L1 + T x_2 L2 + ... + T x_d Ld.
%
%    where T is a complex tensor of size m_1 x ... x m_d, while Lmu is a
%    complex matrix of size m_{mu} x m_{mu}.
%
%    In both cases, if the entry corresponding to the mu-th matrix is empty,
%    then the associated mu-mode product is skipped.
%
%    [CCZ21] M. Caliari, F. Cassini, and F. Zivcovich,
%            A mu-mode BLAS approach for multidimensional tensor-structured
%            problems, Submitted 2021
  if (nargin < 2)
    error('Not enough input arguments.');
  end
  if (iscell(varargin{1}))
    varargin = varargin{1};
  end
  murange = 1:length(varargin);
  murange = murange(~cellfun(@isempty,varargin));
  lmu = length(murange);
  if (lmu == 0)
    error('Not enough non-empty input arguments.');
  end
  sT = size(T);
  kv = mump(T, varargin{murange(1)}, murange(1));
  for mu = murange(2:lmu)
    kv = kv + mump(T, varargin{mu}, mu);
  end
end
%!test % correctness 1d
%! T = randn(2,1);
%! A{1} = randn(2);
%! M = A{1};
%! ref = M*T(:);
%! out1 = kronsumv(T,A);
%! assert(ref,out1(:),1e-10)
%! out2 = kronsumv(T,A{1});
%! assert(ref,out2(:),1e-10)
%!test % correctness 3d
%! T = randn(2,3);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! I{1} = eye(2);
%! I{2} = eye(3);
%! M = kron(I{2},A{1}) + kron(A{2},I{1});
%! ref = M*T(:);
%! out1 = kronsumv(T,A);
%! assert(ref,out1(:),1e-10)
%! out2 = kronsumv(T,A{1},A{2});
%! assert(ref,out2(:),1e-10)
%!test % correctness 3d
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! I{1} = eye(2);
%! I{2} = eye(3);
%! I{3} = eye(4);
%! M = kron(I{3},kron(I{2},A{1})) + kron(I{3},kron(A{2},I{1})) + ...
%!     kron(A{3},kron(I{2},I{1}));
%! ref = M*T(:);
%! out1 = kronsumv(T,A);
%! assert(ref,out1(:),1e-10)
%! out2 = kronsumv(T,A{1},A{2},A{3});
%! assert(ref,out2(:),1e-10)
%!test
%! T = randn(3,4,5);
%! A = randn(4,3);
%! S = kronsumv(T,A);
%! ref = mump(T,A,1);
%! assert(ref,S)
%!test
%! T = randn(4,3,2);
%! A = randn(4);
%! B = randn(3);
%! C = randn(2);
%! S = kronsumv(T,[],B,C);
%! ref = mump(T,B,2) + mump(T,C,3);
%! assert(ref,S);
%! S = kronsumv(T,A,[],C);
%! ref = mump(T,A,1) + mump(T,C,3);
%! assert(ref,S);
%! S = kronsumv(T,A,B,[]);
%! ref = mump(T,A,1) + mump(T,B,2);
%! assert(ref,S);
%! S = kronsumv(T,[],[],C);
%! ref = mump(T,C,3);
%! assert(ref,S);
%! S = kronsumv(T,[],B,[]);
%! ref = mump(T,B,2);
%! assert(ref,S);
%! S = kronsumv(T,A,[],[]);
%! ref = mump(T,A,1);
%! assert(ref,S);
%!error
%! kronsumv();
%!error
%! kronsumv([],[]);
%!error
%! T = randn(3,4,5);
%! kronsumv(T,[],[],[]);
%!error
%! T = randn(3,4,5);
%! kronsumv(T);
