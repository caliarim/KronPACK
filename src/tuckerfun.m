function T = tuckerfun(T,varargin)
% TUCKERFUN Tucker operator with functions.
%    S = TUCKERFUN(T, L) computes the Tucker operator
%
%       S = T x_1 L{1} x_2 L{2} x_3 ... x_d L{d}.
%
%    Here T is a complex tensor of size m_1 x ... x m_d, L a cell array
%    of functions which act on columns of a matrix and x_mu denotes the
%    mu-mode action.
%
%    S = TUCKERFUN(T, L1, L2, ..., Ld) computes the Tucker operator
%
%       S = T x_1 L1 x_2 L2 x_3 ... x_d Ld.
%
%    Here T is a complex tensor of size m_1 x ... x m_d, while Lmu is a
%    function which acts on columns of a matrix.
%
%    In both cases, if the entry corresponding to the mu-th function is empty,
%    then the associated mu-mode action is skipped.
%
%    [CCZ23] M. Caliari, F. Cassini, and F. Zivcovich,
%            A mu-mode BLAS approach for multidimensional tensor-structured
%            problems, NUMERICAL ALGORITHMS 92, 2483-2508 (2023)
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
    T = varargin{1}(reshape(T, sT(1), []));
    sT(1) = size(T, 1);
    T = reshape(T, sT);
    mur = mur(2:lmu);
    lmu = lmu-1;
  end
  if lmu > 0
    T = permute(T, [mur(1), 1:(mur(1)-1), (mur(1)+1):lT]);
    for mu = 1:(lmu-1)
      T = varargin{mur(mu)}(reshape(T, sT(mur(mu)), []));
      sT(mur(mu)) = size(T, 1);
      T = permute(reshape(T, sT([mur(mu), 1:(mur(mu)-1), (mur(mu)+1):lT])), ...
      [mur(mu+1), 2:mur(mu), 1, (mur(mu)+1):(mur(mu+1)-1), (mur(mu+1)+1):lT]);
    end
    T = varargin{mur(lmu)}(reshape(T, sT(mur(lmu)), []));
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
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! Afun{3} = @(X) A{3}*X;
%! assert(tuckerfun(T,Afun),tuckerfun(T,Afun{1},Afun{2},Afun{3}),1e-13)
%! % Function matrix--matrix
%!test % 1d
%! T = randn(2,1);
%! A = randn(3,2);
%! Afun = @(X) A*X;
%! out = tuckerfun(T,Afun);
%! ref = A*T;
%! assert(out,ref,1e-13)
%!test % 2d
%! T = randn(2,3);
%! A{1} = randn(3,2);
%! A{2} = randn(4,3);
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! out = tuckerfun(T,Afun);
%! ref = A{1}*T*A{2}.';
%! assert(out,ref,1e-13)
%!test % 2d complex
%! T = randn(2,3)+1i*randn(2,3);
%! A{1} = randn(3,2)+1i*randn(3,2);
%! A{2} = randn(4,3)+1i*randn(4,3);
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! out = tuckerfun(T,Afun);
%! ref = A{1}*T*A{2}.';
%! assert(out,ref,1e-13)
%!test % 3d
%! T = randn(2,3,4);
%! A{1} = randn(3,2);
%! A{2} = randn(4,3);
%! A{3} = randn(5,4);
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! Afun{3} = @(X) A{3}*X;
%! out = tuckerfun(T,Afun);
%! ref = tucker(T,A);
%! assert(out,ref,1e-13)
%!test % 4d
%! T = randn(2,3,4,5);
%! A{1} = randn(3,2);
%! A{2} = randn(4,3);
%! A{3} = randn(5,4);
%! A{4} = randn(6,5);
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! Afun{3} = @(X) A{3}*X;
%! Afun{4} = @(X) A{4}*X;
%! out = tuckerfun(T,Afun);
%! ref = tucker(T,A);
%! assert(out,ref,1e-13)
%!test % tensorization
%! A{1} = randn(2,1);
%! A{2} = randn(3,1);
%! A{3} = randn(4,1);
%! A{4} = randn(5,1);
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! Afun{3} = @(X) A{3}*X;
%! Afun{4} = @(X) A{4}*X;
%! out = tuckerfun(1,Afun);
%! ref = tensorize(A);
%! assert(out,ref,1e-13)
%!test %tensor with implicit last dimension
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! A{4} = randn(5,1);
%! Afun{1} = @(X) A{1}*X;
%! Afun{2} = @(X) A{2}*X;
%! Afun{3} = @(X) A{3}*X;
%! Afun{4} = @(X) A{4}*X;
%! out = tuckerfun(T,Afun);
%! ref = tucker(T,A);
%! assert(out,ref,1e-13)
%! % Function fft
%!test
%! T = randn(4,6,8);
%! Afun = @(X) fft(X);
%! out = tuckerfun(T,Afun,Afun,Afun);
%! ref = fftn(T);
%! assert(out,ref,1e-13)
%! % Function transpose
%!test
%! T = randn(2,3,4);
%! A{1} = randn(2,3);
%! A{2} = randn(3,4);
%! A{3} = randn(4,5);
%! Afun{1} = @(X) A{1}'*X;
%! Afun{2} = @(X) A{2}'*X;
%! Afun{3} = @(X) A{3}'*X;
%! out = tuckerfun(T,Afun);
%! ref = ttucker(T,A);
%! assert(out,ref,1e-13)
%! % Function inverse
%!test
%! T = randn(2,3,4);
%! A{1} = randn(2);
%! A{2} = randn(3);
%! A{3} = randn(4);
%! Afun{1} = @(X) A{1}\X;
%! Afun{2} = @(X) A{2}\X;
%! Afun{3} = @(X) A{3}\X;
%! out = tuckerfun(T,Afun);
%! ref = itucker(T,A);
%! assert(out,ref,1e-13)
%!test % Jump some modes
%! T = randn(2,3,4,5);
%! A1 = randn(3,2);
%! A2 = randn(4,3);
%! A3 = randn(5,4);
%! A4 = randn(6,5);
%! A1f = @(x) A1*x;
%! A2f = @(x) A2*x;
%! A3f = @(x) A3*x;
%! A4f = @(x) A4*x;
%! out = tuckerfun(T,[],A2f,A3f,A4f);
%! ref = tucker(T,[],A2,A3,A4);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,A1f,[],A3f,A4f);
%! ref = tucker(T,A1,[],A3,A4);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,A1f,A2f,[],A4f);
%! ref = tucker(T,A1,A2,[],A4);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,A1f,A2f,A3f,[]);
%! ref = tucker(T,A1,A2,A3,[]);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,[],A2f,A3f,[]);
%! ref = tucker(T,[],A2,A3,[]);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,A1f,[],[],A4f);
%! ref = tucker(T,A1,[],[],A4);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,A1f,[],A3f,[]);
%! ref = tucker(T,A1,[],A3,[]);
%! assert(out,ref,1e-13)
%! out = tuckerfun(T,[],[],A3f,[]);
%! ref = tucker(T,[],[],A3,[]);
%! assert(out,ref,1e-13)
%!test
%! T = randn(2,3,4,5);
%! A{1} = [];
%! A{2} = randn(3);
%! A{3} = [];
%! A{4} = randn(5);
%! Af{1} = [];
%! Af{2} = @(x) A{2}*x;
%! Af{3} = [];
%! Af{4} = @(x) A{4}*x;
%! ref = tucker(T,A);
%! out = tuckerfun(T,Af);
%! assert(ref,out,1e-13)
%!error
%! tuckerfun();
%!error
%! tuckerfun(randn(3));
%!error
%! tuckerfun(randn(3),[]);
