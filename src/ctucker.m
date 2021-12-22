function T = ctucker(T,varargin)
% CTUCKER Tucker operator with conjugate matrices.
%    S = CTUCKER(T, L) computes the Tucker operator
%
%       S = T x_1 conj(L{1}) x_2 conj(L{2}) x_3 ... x_d conj(L{d}).
%
%    Here T is a complex tensor of size m_1 x ... x m_d, L a cell array
%    of complex matrices (L{mu} of size n_{mu} x m_{mu}) and x_mu denotes
%    the mu-mode product.
%
%    S = CTUCKER(T, L1, L2, ..., Ld) computes the Tucker operator
%
%       S = T x_1 conj(L1) x_2 conj(L2) x_3 ... x_d conj(Ld).
%
%    Here T is a complex tensor of size m_1 x ... x m_d, while Lmu is a
%    complex matrix of size n_{mu} x m_{mu}.
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
  if(mur(1) == 1)
    T = conj(varargin{1})*reshape(T, sT(1), []);
    sT(1) = size(T, 1);
    T = reshape(T, sT);
    mur = mur(2:lmu);
    lmu = lmu-1;
  end
  if lmu > 0
    T = permute(T, [mur(1), 1:(mur(1)-1), (mur(1)+1):lT]);
    for mu = 1:(lmu-1)
      T = conj(varargin{mur(mu)})*reshape(T, sT(mur(mu)), []);
      sT(mur(mu)) = size(T, 1);
      T = permute(reshape(T, sT([mur(mu), 1:(mur(mu)-1), (mur(mu)+1):lT])), ...
      [mur(mu+1), 2:mur(mu), 1, (mur(mu)+1):(mur(mu+1)-1), (mur(mu+1)+1):lT]);
    end
    T = conj(varargin{mur(lmu)})*reshape(T, sT(mur(lmu)), []);
    sT(mur(lmu)) = size(T, 1);
    T = ipermute(reshape(T, sT([mur(lmu), 1:(mur(lmu)-1), (mur(lmu)+1):lT])), ...
        [mur(lmu), 1:(mur(lmu)-1), (mur(lmu)+1):lT]);
  end
end
%!test % different input form
%! T = randn(2,3,4)+1i*randn(2,3,4);
%! A{1} = randn(2)+1i*randn(2);
%! A{2} = randn(3)+1i*randn(3);
%! A{3} = randn(4)+1i*randn(4);
%! assert(ctucker(T,A),ctucker(T,A{1},A{2},A{3}))
%!test % 1d
%! T = randn(2,1)+1i*randn(2,1);
%! A = randn(3,2)+1i*randn(3,2);
%! out = ctucker(T,A);
%! ref = conj(A)*T;
%! assert(out,ref)
%!test % 2d
%! T = randn(2,3)+1i*randn(2,3);
%! A{1} = randn(3,2)+1i*randn(3,2);
%! A{2} = randn(4,3)+1i*randn(4,3);
%! out = ctucker(T,A);
%! ref = conj(A{1})*T*A{2}';
%! assert(out,ref)
%!test % 3d
%! T = randn(2,3,4)+1i*randn(2,3,4);
%! A{1} = randn(3,2)+1i*randn(3,2);
%! A{2} = randn(4,3)+1i*randn(4,3);
%! A{3} = randn(5,4)+1i*randn(5,4);
%! out = ctucker(T,A);
%! ref = mump(mump(mump(T,conj(A{1}),1),conj(A{2}),2),conj(A{3}),3);
%! assert(out,ref)
%!test % 4d
%! T = randn(2,3,4,5) +1i*randn(2,3,4,5);
%! A{1} = randn(3,2)+1i*randn(3,2);
%! A{2} = randn(4,3)+1i*randn(4,3);
%! A{3} = randn(5,4)+1i*randn(5,4);
%! A{4} = randn(6,5)+1i*randn(6,5);
%! out = ctucker(T,A);
%! ref = mump(mump(mump(mump(T,conj(A{1}),1),conj(A{2}),2),conj(A{3}),3),conj(A{4}),4);
%! assert(out,ref)
%!test % tensorization
%! A{1} = randn(2,1)+1i*randn(2,1);
%! A{2} = randn(3,1)+1i*randn(3,1);
%! A{3} = randn(4,1)+1i*randn(4,1);
%! A{4} = randn(5,1)+1i*randn(5,1);
%! out = ctucker(1,A);
%! ref = tensorize(conj(A{1}),conj(A{2}),conj(A{3}),conj(A{4}));
%! assert(out,ref)
%!test %tensor with implicit last dimension
%! T = randn(2,3,4)+1i*randn(2,3,4);
%! A{1} = randn(2)+1i*randn(2);
%! A{2} = randn(3)+1i*randn(3);
%! A{3} = randn(4)+1i*randn(4);
%! A{4} = randn(5,1)+1i*randn(5,1);
%! out = ctucker(T,A);
%! ref = mump(mump(mump(mump(T,conj(A{1}),1),conj(A{2}),2),conj(A{3}),3),conj(A{4}),4);
%! assert(out,ref)
%!test % Jump some modes
%! T = randn(2,3,4,5)+1i*randn(2,3,4,5);
%! A1 = randn(3,2)+1i*randn(3,2);
%! A2 = randn(4,3)+1i*randn(4,3);
%! A3 = randn(5,4)+1i*randn(5,4);
%! A4 = randn(6,5)+1i*randn(6,5);
%! out = ctucker(T,[],A2,A3,A4);
%! ref = mump(mump(mump(T,conj(A2),2),conj(A3),3),conj(A4),4);
%! assert(out,ref)
%! out = ctucker(T,A1,[],A3,A4);
%! ref = mump(mump(mump(T,conj(A1),1),conj(A3),3),conj(A4),4);
%! assert(out,ref)
%! out = ctucker(T,A1,A2,[],A4);
%! ref = mump(mump(mump(T,conj(A1),1),conj(A2),2),conj(A4),4);
%! assert(out,ref)
%! out = ctucker(T,A1,A2,A3,[]);
%! ref = mump(mump(mump(T,conj(A1),1),conj(A2),2),conj(A3),3);
%! assert(out,ref)
%! out = ctucker(T,[],A2,A3,[]);
%! ref = mump(mump(T,conj(A2),2),conj(A3),3);
%! assert(out,ref)
%! out = ctucker(T,A1,[],[],A4);
%! ref = mump(mump(T,conj(A1),1),conj(A4),4);
%! assert(out,ref)
%! out = ctucker(T,A1,[],A3,[]);
%! ref = mump(mump(T,conj(A1),1),conj(A3),3);
%! assert(out,ref)
%! out = ctucker(T,[],[],A3,[]);
%! ref = mump(T,conj(A3),3);
%! assert(out,ref)
%!error
%! ctucker();
%!error
%! ctucker(randn(3));
%!error
%! ctucker(randn(3),[]);
