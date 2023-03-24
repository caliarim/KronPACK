% Example of multivariate interpolation (see [CCZ23, Sec. 4.3])
%
% Function: f(x1,x2,x3,x4,x5)=1/(1+16*(x1^2+x2^2+x3^2+x4^2+x5^2))
% Domain: [a1,b1]x[a2,b2]x[a3,b3]x[a4,b4]x[a5,b5]=[-1,1]^5
% Method: Second barycentric Lagrange interpolation
% Interpolation nodes: mbold Chebyshev points
% Evaluation points: nbold uniformely distributed
%
% [CCZ23] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional tensor-structured
%         problems, NUMERICAL ALGORITHMS 92, 2483-2508 (2023)

clear all
addpath('../src')
fprintf('---- Multivariate interpolation ----\n')
d = 5;
f = @(x1, x2, x3, x4, x5) 1./(1+16*(x1.^2+x2.^2+x3.^2+x4.^2+x5.^2));
a = -ones(1, d);
b = ones(1, d);
nbold = 35*ones(1, d);
for mu = 1:d
  x{mu} = linspace(a(mu), b(mu), nbold(mu))';
end
[X{1:d}] = ndgrid(x{1:d});
F_exact = f(X{1:d});
F_exact_norm = max(abs(F_exact(:)));
mrange = 5:10:45;
counter = 0;
for m = mrange
  counter = counter+1;
  mbold = m*ones(1, d);
  for mu = 1:d
    kmu = (1:mbold(mu))';
    xi{mu} = (a(mu)+b(mu))/2+(b(mu)-a(mu))/2*cos((2*kmu-1)/(2*mbold(mu))*pi);
    w{mu} = sin((2*kmu-1)/(2*mbold(mu))*pi).*(-1).^(kmu+1);
    for imu = 1:mbold(mu)
      L{mu}(:,imu) = w{mu}(imu)./(x{mu}-xi{mu}(imu));
    end
    L{mu} = diag(1./(sum(L{mu}, 2)))*L{mu};
  end
  [XI{1:d}] = ndgrid(xi{1:d});
  F = f(XI{1:d});
  P = tucker(F, L);
  err = P-F_exact;
  rel_err_norm = max(abs(err(:)))/F_exact_norm;
  fprintf('Lagrange interpolation error for m = %3d: %.2e\n', m, rel_err_norm)
  err_mrange(counter)= rel_err_norm;
end
fprintf(' \n');
rmpath('../src')
K = 1/4+sqrt(17/16);
figure;
semilogy(mrange, err_mrange, '*',...
         mrange, err_mrange(end)/K^(-mrange(end))*K.^(-mrange))
legend('Relative error norm', 'Theoretical decay estimate')
xlabel('m')
ylabel('Error')
title(sprintf('Multivariate interpolation (d = %d)',d))
drawnow
