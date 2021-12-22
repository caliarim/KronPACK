function T = itucker(T,varargin)
% ITUCKER Tucker operator with inverse matrices.
%    S = ITUCKER(T, L) computes the Tucker operator
%
%       S = T x_1 L{1}^{-1} x_2 L{2}^{-1} x_3 ... x_d L{d}^{-1}
%
%    without explicitly computing the inverse matrices. Here T is a complex
%    tensor of size m_1 x ... x m_d, L a cell array of complex matrices
%    (L{mu} of size m_{mu} x n_{mu}) and x_mu denotes the mu-mode product.
%
%    S = ITUCKER(T, L1, L2, ..., Ld) computes the Tucker operator
%
%       S = T x_1 L1^{-1} x_2 L2^{-1} x_3 ... x_d Ld^{-1}.
%
%    without explicitly computing the inverse matrices. Here T is a complex
%    tensor of size m_1 x ... x m_d, while Lmu is a complex matrix of size
%    m_{mu} x n_{mu}.
%
%    In both cases, if m_{mu} ~= n_{mu} a least square solution of the
%    corresponding linear system is computed along the mu-th direction.
%    Moreover, if the entry corresponding to the mu-th matrix is empty,
%    the associated mu-mode product is skipped.
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
  if(mur(1) == 1)
    T = varargin{1}\reshape(T, sT(1), []);
    sT(1) = size(T, 1);
    T = reshape(T, sT);
    mur = mur(2:lmu);
    lmu = lmu-1;
  end
  if lmu > 0
    T = permute(T, [mur(1), 1:(mur(1)-1), (mur(1)+1):lT]);
    for mu = 1:(lmu-1)
      T = varargin{mur(mu)}\reshape(T, sT(mur(mu)), []);
      sT(mur(mu)) = size(T, 1);
      T = permute(reshape(T, sT([mur(mu), 1:(mur(mu)-1), (mur(mu)+1):lT])), ...
      [mur(mu+1), 2:mur(mu), 1, (mur(mu)+1):(mur(mu+1)-1), (mur(mu+1)+1):lT]);
    end
    T = varargin{mur(lmu)}\reshape(T, sT(mur(lmu)), []);
    sT(mur(lmu)) = size(T, 1);
    T = ipermute(reshape(T, sT([mur(lmu), 1:(mur(lmu)-1), (mur(lmu)+1):lT])), ...
        [mur(lmu), 1:(mur(lmu)-1), (mur(lmu)+1):lT]);
  end
end
%!test % different input form
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! assert(itucker(T,A),itucker(T,A{1},A{2},A{3}))
%!test
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! out = itucker(T,A);
%! ref = tucker(T,inv(A{1}),inv(A{2}),inv(A{3}));
%! assert(out,ref,1e-10)
%!test % 1d
%! T = randn(2);
%! A = randn(2);
%! out = itucker(T,A);
%! ref = A\T;
%! assert(out,ref,1e-10)
%!test % 2d
%! T = randn(2,3);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! out = itucker(T,A);
%! ref = mump(mump(T,inv(A{1}),1),inv(A{2}),2);
%! assert(out,ref,1e-10)
%!test % 2d complex
%! T = randn(2,3)+1i*randn(2,3);
%! A{1} = randn(2)+1i*randn(2);
%! A{2} = randn(3)+1i*randn(3);
%! out = itucker(T,A);
%! ref = mump(mump(T,inv(A{1}),1),inv(A{2}),2);
%! assert(out,ref,1e-10)
%!test % 3d
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! out = itucker(T,A);
%! ref = mump(mump(mump(T,inv(A{1}),1),inv(A{2}),2),inv(A{3}),3);
%! assert(out,ref,1e-10)
%!test
%! T = randn(5,4);
%! A = randn(5,3);
%! B = randn(4,2);
%! ref = (B\(A\T)')';
%!test
%! T = randn(3,4);
%! A = randn(3,5);
%! B = randn(4,2);
%! C = randn(1,6);
%! ref = ipermute(reshape(C\(reshape(permute((B\(A\T)')',[3,1,2]),1,5*2)),...
%!       [6,5,2]),[3,1,2]);
%! assert(ref,itucker(T,A,B,C))
%!test % Jump some modes
%! T = randn(2,3,4,5);
%! A1 = randn(2);
%! A2 = randn(3);
%! A3 = randn(4);
%! A4 = randn(5);
%! out = itucker(T,[],A2,A3,A4);
%! ref = mump(mump(mump(T,inv(A2),2),inv(A3),3),inv(A4),4);
%! assert(out,ref,1e-10)
%! out = itucker(T,A1,[],A3,A4);
%! ref = mump(mump(mump(T,inv(A1),1),inv(A3),3),inv(A4),4);
%! assert(out,ref,1e-10)
%! out = itucker(T,inv(A1),inv(A2),[],inv(A4));
%! ref = mump(mump(mump(T,A1,1),A2,2),A4,4);
%! assert(out,ref,1e-10)
%! out = itucker(T,A1,A2,A3,[]);
%! ref = mump(mump(mump(T,inv(A1),1),inv(A2),2),inv(A3),3);
%! assert(out,ref,1e-10)
%! out = itucker(T,[],A2,A3,[]);
%! ref = mump(mump(T,inv(A2),2),inv(A3),3);
%! assert(out,ref,1e-10)
%! out = itucker(T,A1,[],[],A4);
%! ref = mump(mump(T,inv(A1),1),inv(A4),4);
%! assert(out,ref,1e-10)
%! out = itucker(T,A1,[],A3,[]);
%! ref = mump(mump(T,inv(A1),1),inv(A3),3);
%! assert(out,ref,1e-10)
%! out = itucker(T,[],[],A3,[]);
%! ref = mump(T,inv(A3),3);
%! assert(out,ref,1e-10)
%!error
%! itucker();
%!error
%! itucker(randn(2));
%!error
%! itucker(randn(2),[]);
