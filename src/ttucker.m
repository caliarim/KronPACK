function T = ttucker(T,varargin)
% TTUCKER Tucker operator with transposed matrices.
%    S = TTUCKER(T, L) computes the Tucker operator
%
%       S = T x_1 L{1}.' x_2 L{2}.' x_3 ... x_d L{d}.'
%
%    without explicit transposition of the matrices.
%    Here T is a complex tensor of size m_1 x ... x m_d, L a cell array
%    of complex matrices (L{mu} of size m_{mu} x n_{mu}) and x_mu denotes
%    the mu-mode product.
%
%    S = TTUCKER(T, L1, L2, ..., Ld) computes the Tucker operator
%
%       S = T x_1 L1.' x_2 L2.' x_3 ... x_d Ld.'
%
%    without explicit transposition of the matrices.
%    Here T is a complex tensor of size m_1 x ... x m_d, while Lmu is a
%    complex matrix of size m_{mu} x n_{mu}.
%
%    In both cases, if the entry corresponding to the mu-th matrix is empty,
%    then the associated mu-mode product is skipped.
%
%    [CCZ22] M. Caliari, F. Cassini, and F. Zivcovich,
%            A mu-mode BLAS approach for multidimensional tensor-structured
%            problems, Submitted 2022
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
  if (mur(1) == 1) && (mur(lmu) == lT)
    T = varargin{1}.'*reshape(T, sT(1), []);
    sT(1) = size(T, 1);
    T = reshape(T, [], sT(mur(lmu)))*varargin{mur(lmu)};
    sT(mur(lmu)) = size(T, 2);
    T = reshape(T, sT);
    mur = mur(2:lmu-1);
    lmu = lmu-2;
  elseif (mur(1) == 1) && (mur(lmu) ~= lT)
    T = varargin{1}.'*reshape(T, sT(1), []);
    sT(1) = size(T, 1);
    T = reshape(T, sT);
    mur = mur(2:lmu);
    lmu = lmu-1;
  elseif (mur(1) ~= 1) && (mur(lmu) == lT)
    T = reshape(T, [], sT(mur(lmu)))*varargin{mur(lmu)};
    sT(mur(lmu)) = size(T, 2);
    T = reshape(T, sT);
    mur = mur(1:lmu-1);
    lmu = lmu-1;
  end
  if (lmu > 0)
    T = permute(T, [1:(mur(1)-1), (mur(1)+1):lT, mur(1)]);
    for mu = 1:(lmu-1)
      T = reshape(T, [], sT(mur(mu)))*varargin{mur(mu)};
      sT(mur(mu)) = size(T, 2);
      T = permute(reshape(T, sT([1:(mur(mu)-1), (mur(mu)+1):lT, mur(mu)])), ...
      [1:(mur(mu)-1), lT, mur(mu):(mur(mu+1)-2), mur(mu+1):(lT-1), mur(mu+1)-1]);
    end
    T = reshape(T, [],sT(mur(lmu)))*varargin{mur(lmu)};
    sT(mur(lmu)) = size(T, 2);
    T = ipermute(reshape(T, sT([1:(mur(lmu)-1), (mur(lmu)+1):lT, mur(lmu)])), ...
        [1:(mur(lmu)-1), (mur(lmu)+1):lT, mur(lmu)]);
  end
end
%!test % different input form
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! assert(ttucker(T,A),ttucker(T,A{1},A{2},A{3}))
%!test % 1d
%! T = randn(2,1);
%! A = randn(2,3);
%! out = ttucker(T,A);
%! ref = A.'*T;
%! assert(out,ref,1e-13)
%!test % 2d
%! T = randn(2,3);
%! A{1} = randn(2,3);
%! A{2} = randn(3,4);
%! out = ttucker(T,A);
%! ref = A{1}.'*T*A{2};
%! assert(out,ref,1e-13)
%!test % 3d
%! T = randn(2,3,4);
%! A{1} = randn(2,3);
%! A{2} = randn(3,4);
%! A{3} = randn(4,5);
%! out = ttucker(T,A);
%! ref = mump(mump(mump(T,A{1}.',1),A{2}.',2),A{3}.',3);
%! assert(out,ref,1e-13)
%!test % 4d
%! T = randn(2,3,4,5);
%! A{1} = randn(2,3);
%! A{2} = randn(3,4);
%! A{3} = randn(4,5);
%! A{4} = randn(5,6);
%! out = ttucker(T,A);
%! ref = mump(mump(mump(mump(T,A{1}.',1),A{2}.',2),A{3}.',3),A{4}.',4);
%! assert(out,ref,1e-13)
%!test % tucker
%! T = randn(2,3,4,5);
%! A{1} = randn(2,3);
%! A{2} = randn(3,4);
%! A{3} = randn(4,5);
%! A{4} = randn(5,6);
%! out = ttucker(T,A);
%! ref = tucker(T,A{1}.',A{2}.',A{3}.',A{4}.');
%! assert(out,ref,1e-13)
%!test % tensorization
%! A{1} = randn(1,2);
%! A{2} = randn(1,3);
%! A{3} = randn(1,4);
%! A{4} = randn(1,5);
%! out = ttucker(1,A);
%! ref = tensorize(A{1}.',A{2}.',A{3}.',A{4}.');
%! assert(out,ref,1e-13)
%!test %tensor with implicit last dimension
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! A{4} = randn(1,5);
%! out = ttucker(T,A);
%! ref = tucker(T,A{1}.',A{2}.',A{3}.',A{4}.');
%! assert(out,ref,1e-13)
%!test % complex
%! T = randn(2,3,4)+1i*randn(2,3,4);
%! A{1} = randn(2,3)+1i*randn(2,3);
%! A{2} = randn(3,4)+1i*randn(3,4);
%! A{3} = randn(4,5)+1i*randn(4,5);
%! out = ttucker(T,A);
%! ref = tucker(T,A{1}.',A{2}.',A{3}.');
%! assert(out,ref,1e-13)
%!test % Jump some modes
%! T = randn(2,3,4,5);
%! A1 = randn(2,3);
%! A2 = randn(3,4);
%! A3 = randn(4,5);
%! A4 = randn(5,6);
%! out = ttucker(T,[],A2,A3,A4);
%! ref = mump(mump(mump(T,A2.',2),A3.',3),A4.',4);
%! assert(out,ref,1e-13)
%! out = ttucker(T,A1,[],A3,A4);
%! ref = mump(mump(mump(T,A1.',1),A3.',3),A4.',4);
%! assert(out,ref,1e-13)
%! out = ttucker(T,A1,A2,[],A4);
%! ref = mump(mump(mump(T,A1.',1),A2.',2),A4.',4);
%! assert(out,ref,1e-13)
%! out = ttucker(T,A1,A2,A3,[]);
%! ref = mump(mump(mump(T,A1.',1),A2.',2),A3.',3);
%! assert(out,ref,1e-13)
%! out = ttucker(T,[],A2,A3,[]);
%! ref = mump(mump(T,A2.',2),A3.',3);
%! assert(out,ref,1e-13)
%! out = ttucker(T,A1,[],[],A4);
%! ref = mump(mump(T,A1.',1),A4.',4);
%! assert(out,ref,1e-13)
%! out = ttucker(T,A1,[],A3,[]);
%! ref = mump(mump(T,A1.',1),A3.',3);
%! assert(out,ref,1e-13)
%! out = ttucker(T,[],[],A3,[]);
%! ref = mump(T,A3.',3);
%! assert(out,ref,1e-13)
%!error
%! ttucker();
%!error
%! ttucker(randn(3));
%!error
%! ttucker(randn(3),[]);
