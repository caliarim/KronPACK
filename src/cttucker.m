function T = cttucker(T,varargin)
% CTTUCKER Tucker operator with conjugate transposed matrices.
%    S = CTTUCKER(T, L) computes the Tucker operator
%
%       S = T x_1 L{1}' x_2 L{2}' x_3 ... x_d L{d}'
%
%    without explicit transposition.
%    Here T is a complex tensor of size m_1 x ... x m_d, L a cell array
%    of complex matrices (L{mu} of size m_{mu} x n_{mu}) and x_mu denotes
%    the mu-mode product.
%
%    S = CTTUCKER(T, L1, L2, ..., Ld) computes the Tucker operator
%
%       S = T x_1 L1' x_2 L2' x_3 ... x_d Ld'
%
%    without explicit transposition.
%    Here T is a complex tensor of size m_1 x ... x m_d, while Lmu is a
%    complex matrix of size m_{mu} x n_{mu}.
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
  if (iscell (varargin{1}))
    varargin = varargin{1};
  end
  eidx = ~cellfun(@isempty, varargin);
  sT = [size(T), ones(1, length(varargin)-find(flip(eidx), 1)+1-length(size(T)))];
  lT = length(sT);
  mur = 1:length(varargin);
  mur = mur(eidx);
  lmu = length(mur);
  if (lmu == 0)
    error('Not enough non-empty input arguments.');
  end
  if (mur(lmu) == lT)
    T = reshape(T, [], sT(mur(lmu)))*conj(varargin{mur(lmu)});
    sT(mur(lmu)) = size(T, 2);
    T = reshape(T, sT);
    mur = mur(1:lmu-1);
    lmu = lmu-1;
  end
  if(lmu > 0)
    T = permute(T, [1:(mur(1)-1), (mur(1)+1):lT, mur(1)]);
    for mu = 1:(lmu-1)
      T = reshape(T, [], sT(mur(mu)))*conj(varargin{mur(mu)});
      sT(mur(mu)) = size(T, 2);
      T = permute(reshape(T, sT([1:(mur(mu)-1), (mur(mu)+1):lT, mur(mu)])), ...
      [1:(mur(mu)-1), lT, mur(mu):(mur(mu+1)-2), mur(mu+1):(lT-1), mur(mu+1)-1]);
    end
    T = reshape(T, [],sT(mur(lmu)))*conj(varargin{mur(lmu)});
    sT(mur(lmu)) = size(T, 2);
    T = ipermute(reshape(T, sT([1:(mur(lmu)-1), (mur(lmu)+1):lT, mur(lmu)])), ...
        [1:(mur(lmu)-1), (mur(lmu)+1):lT, mur(lmu)]);
  end
end
%!test % different input form
%! T = rand*1i + randn(2,3,4);
%! A{1} = rand*1i + randn(2);
%! A{2} = rand*1i +randn(3);
%! A{3} = rand*1i +randn(4);
%! assert(cttucker(T,A),cttucker(T,A{1},A{2},A{3}))
%!test % 1d
%! T = rand*1i +randn(2,1);
%! A = rand*1i +randn(2,3);
%! out = cttucker(T,A);
%! ref = A'*T;
%! assert(out,ref,1e-13)
%!test % 2d
%! T = rand*1i +randn(2,3);
%! A{1} = rand*1i +randn(2,3);
%! A{2} = rand*1i +randn(3,4);
%! out = cttucker(T,A);
%! ref = A{1}'*T*conj(A{2});
%! assert(out,ref,1e-13)
%!test % 3d
%! T = rand*1i +randn(2,3,4);
%! A{1} = rand*1i +randn(2,3);
%! A{2} = rand*1i +randn(3,4);
%! A{3} = rand*1i +randn(4,5);
%! out = cttucker(T,A);
%! ref = mump(mump(mump(T,A{1}',1),A{2}',2),A{3}',3);
%! assert(out,ref,1e-13)
%!test % 3d real
%! T = randn(2,3,4);
%! A{1} = randn(2,3);
%! A{2} = randn(3,4);
%! A{3} = randn(4,5);
%! out = cttucker(T,A);
%! ref = mump(mump(mump(T,A{1}',1),A{2}',2),A{3}',3);
%! assert(out,ref,1e-13)
%!test % 4d
%! T = rand*1i +randn(2,3,4,5);
%! A{1} = rand*1i +randn(2,3);
%! A{2} = rand*1i +randn(3,4);
%! A{3} = rand*1i +randn(4,5);
%! A{4} = rand*1i +randn(5,6);
%! out = cttucker(T,A);
%! ref = mump(mump(mump(mump(T,A{1}',1),A{2}',2),A{3}',3),A{4}',4);
%! assert(out,ref,1e-13)
%!test % tensorization
%! A{1} = randn*1i + randn(1,2);
%! A{2} = randn*1i +randn(1,3);
%! A{3} = randn*1i +randn(1,4);
%! A{4} = randn*1i +randn(1,5);
%! out = cttucker(1,A);
%! ref = tensorize(A{1}',A{2}',A{3}',A{4}');
%! assert(out,ref,1e-13)
%!test %tensor with implicit last dimension
%! T = randn*1i+ randn(2,3,4);
%! A{1} = randn*1i+randn(2);
%! A{2} = randn*1i+randn(3);
%! A{3} = randn*1i+randn(4);
%! A{4} = randn*1i+randn(1,5);
%! out = cttucker(T,A);
%! ref = tucker(T,A{1}',A{2}',A{3}',A{4}');
%! assert(out,ref,1e-13)
%!test % Jump some modes
%! T =randn*1i+ randn(2,3,4,5);
%! A1 = randn*1i+randn(2,3);
%! A2 = randn*1i+randn(3,4);
%! A3 = randn*1i+randn(4,5);
%! A4 = randn*1i+randn(5,6);
%! out = cttucker(T,[],A2,A3,A4);
%! ref = mump(mump(mump(T,A2',2),A3',3),A4',4);
%! assert(out,ref,1e-13)
%! out = cttucker(T,A1,[],A3,A4);
%! ref = mump(mump(mump(T,A1',1),A3',3),A4',4);
%! assert(out,ref,1e-13)
%! out = cttucker(T,A1,A2,[],A4);
%! ref = mump(mump(mump(T,A1',1),A2',2),A4',4);
%! assert(out,ref,1e-13)
%! out = cttucker(T,A1,A2,A3,[]);
%! ref = mump(mump(mump(T,A1',1),A2',2),A3',3);
%! assert(out,ref,1e-13)
%! out = cttucker(T,[],A2,A3,[]);
%! ref = mump(mump(T,A2',2),A3',3);
%! assert(out,ref,1e-13)
%! out = cttucker(T,A1,[],[],A4);
%! ref = mump(mump(T,A1',1),A4',4);
%! assert(out,ref,1e-13)
%! out = cttucker(T,A1,[],A3,[]);
%! ref = mump(mump(T,A1',1),A3',3);
%! assert(out,ref,1e-13)
%! out = cttucker(T,[],[],A3,[]);
%! ref = mump(T,A3',3);
%! assert(out,ref,1e-13)
%!error
%! cttucker();
%!error
%! cttucker(randn(3));
%!error
%! cttucker(randn(3),[]);
