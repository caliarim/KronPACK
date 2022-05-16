function T = tensorize(varargin)
% TENSORIZE Build a d-dimensional array starting from d vectors, d > 1.
%   T = TENSORIZE(V) builds the tensor T from the vectors contained in
%   the cell array V, such that the vector T(:) is
%
%      KRON(V{d}(:), KRON(V{d-1}(:), ..., KRON(V{2}(:), V{1}(:))))
%
%   T = TENSORIZE(V1, V2, ..., Vd) builds the tensor T from the vectors
%   V1, V2, ..., Vd, such that the vector T(:) is
%
%      KRON(Vd(:), KRON(Vd-1(:), ..., KRON(V2(:), V1(:)))).
%
%   If the mu-th input is empty, that direction is skipped.
%
%   TENSORIZE(V) is equivalent to TUCKER(1,W), with W{mu} = V{mu}(:).
%   The latter syntax has to be used in Matlab < R2016b or GNU Octave < 3.6.
%
%   [CCZ22] M. Caliari, F. Cassini, and F. Zivcovich,
%           A mu-mode BLAS approach for multidimensional tensor-structured
%           problems, Submitted 2022
  if (nargin < 1)
    error('Not enough input arguments.');
  end
  if (iscell(varargin{1}))
    varargin = varargin{1};
  end
  murange = 1:length(varargin);
  murange = murange(~cellfun(@isempty,varargin));
  lmurange = length(murange);
  if (lmurange < 2)
    error('Not enough non-empty input arguments.')
  else
    T = reshape(varargin{murange(1)},...
        [ones(1, murange(1)-1), length(varargin{murange(1)}), ones(1,2-murange(1))]);
    for mu = murange(2:lmurange)
      T = T.*reshape(varargin{mu}, [ones(1, mu-1), length(varargin{mu})]);
    end
  end
end
%!test % different input form
%! v1 = randn(2,1); v2 = randn(3,1); v3 = randn(4,1);
%! v1o = v1; v2o = v2'; v3o = reshape(v3,[1,1,4]);
%! ref = kron(v3,kron(v2,v1));
%! V = tensorize(v1,v2,v3);
%! assert(ref,V(:))
%! V2 = tensorize({v1,v2,v3});
%! assert(ref,V2(:))
%! V3 = tensorize(v1o,v2o,v3o);
%! assert(ref,V3(:))
%! V4 = tensorize({v1o,v2o,v3o});
%! assert(ref,V4(:))
%!test % 2d
%! v1 = randn(2,1); v2 = randn(3,1);
%! ref = kron(v2,v1);
%! V = tensorize(v1,v2);
%! assert(ref,V(:))
%!test % 3d
%! v1 = randn(2,1); v2 = randn(3,1); v3 = randn(4,1);
%! ref = kron(v3,kron(v2,v1));
%! V = tensorize(v1,v2,v3);
%! assert(ref,V(:))
%!test % 4d
%! v1 = randn(2,1); v2 = randn(3,1); v3 = randn(4,1); v4 = randn(5,1);
%! ref = kron(v4,kron(v3,kron(v2,v1)));
%! V = tensorize(v1,v2,v3,v4);
%! assert(ref,V(:))
%!test % complex
%! v1 = randn(2,1)+1i*randn(2,1);
%! v2 = randn(3,1)+1i*randn(3,1);
%! v3 = randn(4,1)+1i*randn(4,1);
%! ref = kron(v3,kron(v2,v1));
%! V = tensorize(v1,v2,v3);
%! assert(ref,V(:))
%!test % tucker
%! v1 = randn(2,1); v2 = randn(3,1); v3 = randn(4,1); v4 = randn(5,1);
%! ref = tucker(1,v1,v2,v3,v4);
%! V = tensorize(v1,v2,v3,v4);
%!test % jumping directions
%! v1 = randn(2,1); v2 = randn(3,1);
%! V = tensorize(v1,[],v2);
%! ref = v1.*reshape(v2,[1,1,3]);
%! assert(ref,V)
%! V = tensorize([],v1,v2);
%! ref = v1'.*reshape(v2,[1,1,3]);
%! assert(ref,V)
%! V = tensorize(v1,v2,[]);
%! ref = v1.*v2';
%! assert(ref,V)
%!error % empty
%! V = tensorize([]);
%!error
%! V = tensorize([],[]);
%!error
%! V = tensorize();
%!error
%! V = tensorize(randn(3,1),[]);
%!error
%! V = tensorize([],randn(3,1));
%!error
%! V = tensorize(randn(3,1));
