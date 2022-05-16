% Example of HLF function decomposition (see [CCZ22, Sec. 4.2])
%
% Function:
% f(x1,x2,x3)=x2^2*sin(10*x2)*sin(20*x1)/(sin(2*pi*x3)+2)*exp(-x1^2-2*x2)
% Domain: [a1,b1]x[a2,b2]x[a3,b3]=[-4,4]x[0,11]x[-1,1]
% Method: Hermite-Laguerre-Fourier and Fourier-Fourier-Fourier
% Evaluation points: nbold uniformely distributed
%
% [CCZ22] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional
%         tensor-structured problems, Submitted 2022

clear all
addpath('../src')
disp(sprintf('---- HLF function decomposition ----'))
d = 3;
f = @(x1, x2, x3) x2.^2.*sin(10*x2).*sin(20*x1)./(sin(2*pi*x3)+2).*exp(-x1.^2-2*x2);
alpha = 4;
a = [-4, 0, -1];
b = [4, 11, 1];
nbold = 301*ones(1, d);
tol = 10.^[-1:-1:-5];
mrange_HLF = [45, 31, 8; ...
              49, 45, 16; ...
              53, 59, 24; ...
              57, 75, 32; ...
              69, 105, 38];
mrange_FFF = [58, 52, 8; ...
              62, 76, 16; ...
              64, 136, 24; ...
              66, 250, 32; ...
              68, 300, 38];
for mu = 1:d
  x{mu} = linspace(a(mu), b(mu), nbold(mu)+1)';
  x{mu} = x{mu}(1:nbold(mu));
end
[X{1:d}] = ndgrid(x{1:d});
F_exact = f(X{1:d});
F_exact_norm = max(abs(F_exact(:)));
for idx = 1:size(mrange_HLF,1)
  disp(sprintf('Prescribed accuracy: %.2e', tol(idx)))
  % HLF approach
  mbold_HLF = mrange_HLF(idx,:);
  % Hermite
  beta0(1) = sqrt(2*mbold_HLF(1)+1);
  beta(1) = beta0(1)/b(1);
  [xh, wh] = hermpts(mbold_HLF(1));
  xi{1} = xh/beta(1);
  wxi{1} = exp(-beta(1)^2*xi{1}.^2/2);
  w{1} = wh/beta(1)./wxi{1}'./wxi{1}';
  wx{1} = exp(-beta(1)^2.*x{1}'.^2/2);
  % Laguerre
  [xl, wl] = lagpts(mbold_HLF(2), alpha);
  beta0(2) = 4*mbold_HLF(2)+2*alpha+2;
  beta(2) = beta0(2)/b(2);
  xi{2} = xl/beta(2);
  wxi{2} = exp(-beta(2)*xi{2}/2).*(beta(2)*xi{2}).^(alpha/2);
  w{2} = wl/beta(2)./wxi{2}'./ wxi{2}';
  wx{2} = exp(-beta(2)*x{2}'/2).*(beta(2)*x{2}').^(alpha/2);
  % Fourier
  xf = linspace(a(3), b(3), mbold_HLF(3)+1)';
  xi{3} = xf(1:mbold_HLF(3));
  w{3} = ones(1,mbold_HLF(3));
  W = tensorize(w);
  [XI{1:d}] = ndgrid(xi{1:d});
  F = f(XI{1:d});
  FW = F.*W;
  tic
  PSI{1} = ones(mbold_HLF(1))*sqrt(beta(1))/sqrt(sqrt(pi));
  PSI{1}(1,:) = PSI{1}(1,:).*wxi{1}';
  PSI{1}(2,:) = sqrt(2)*beta(1)*xi{1}'.*PSI{1}(2,:).*wxi{1}';
  for i1 = 3:mbold_HLF(1)
    PSI{1}(i1,:) = (sqrt(2)*beta(1)*xi{1}'.*PSI{1}(i1-1,:)-...
                   sqrt(i1-2)*PSI{1}(i1-2,:))/sqrt(i1-1);
  end
  PSI{2} = ones(mbold_HLF(2))*sqrt(beta(2)/gamma(alpha+1));
  PSI{2}(1,:) = PSI{2}(1,:).*wxi{2}';
  PSI{2}(2,:) = (1+alpha-beta(2)*xi{2}').*PSI{2}(2,:)/sqrt(alpha+1).*wxi{2}';
  for i2 = 3:mbold_HLF(2)
    PSI{2}(i2,:) = (2*i2-3+alpha-beta(2)*xi{2}')/sqrt((i2+alpha-1)*...
                   (i2-1)).*PSI{2}(i2-1,:)-sqrt((i2-2+alpha)*(i2-2)/...
                   (i2+alpha-1)/(i2-1))*PSI{2}(i2-2,:);
  end
  PSIfun{1} = @(f) PSI{1}*f;
  PSIfun{2} = @(f) PSI{2}*f;
  PSIfun{3} = @(f) f;
  Fhat = tuckerfun(FW, PSIfun);
  PHI{1} = ones(nbold(1), mbold_HLF(1))*sqrt(beta(1))/sqrt(sqrt(pi));
  PHI{1}(:,1) = PHI{1}(:,1).*wx{1}';
  PHI{1}(:,2) = sqrt(2)*beta(1)*x{1}.*PHI{1}(:,2).*wx{1}';
  for i1 = 3:mbold_HLF(1)
    PHI{1}(:,i1) = (sqrt(2)*beta(1)*x{1}.*PHI{1}(:,i1-1)-...
                    sqrt(i1-2)*PHI{1}(:,i1-2))/sqrt(i1-1);
  end
  PHI{2} = ones(nbold(2), mbold_HLF(2))*sqrt(beta(2)/gamma(alpha+1));
  PHI{2}(:,1) = PHI{2}(:,1).*wx{2}';
  PHI{2}(:,2) = (1+alpha-beta(2)*x{2}).*PHI{2}(:,2)/sqrt(alpha+1).*wx{2}';
  for i2 = 3:mbold_HLF(2)
    PHI{2}(:,i2) = (2*i2-3+alpha-beta(2)*x{2})/sqrt((i2+alpha-1)*(i2-1)).*...
                  PHI{2}(:,i2-1)-(i2-2+alpha)*...
                                 sqrt((i2-2)/(i2+alpha-2)/(i2+alpha-1)/...
                                      (i2-1))*PHI{2}(:,i2-2);
  end
  PHIfun{1} = @(f) PHI{1}*f;
  PHIfun{2} = @(f) PHI{2}*f;
  PHIfun{3} = @(f) interpft(f, nbold(3));
  Ftilde = tuckerfun(Fhat, PHIfun);
  HLF_elapsed = toc;
  HLF_err = Ftilde-F_exact;
  HLF_rel_err_norm = max(abs(HLF_err(:)))/F_exact_norm;
  disp(sprintf('mbold_HLF = (%i,%i,%i)', mbold_HLF(1), mbold_HLF(2), mbold_HLF(3)))
  disp(sprintf('HLF error: %.2e', HLF_rel_err_norm))
  disp(sprintf('HLF elapsed time: %.2e', HLF_elapsed))
  % FFF approach
  mbold_FFF = mrange_FFF(idx,:);
  for mu = 1:d
    xi{mu} = linspace(a(mu), b(mu), mbold_FFF(mu)+1)';
    xi{mu} = xi{mu}(1:mbold_FFF(mu));
  end
  [XI{1:d}] = ndgrid(xi{1:d});
  F = f(XI{1:d});
  tic
  Ftilde = interpft(interpft(interpft(F, nbold(1), 1), nbold(2) , 2), nbold(3) ,3);
  FFF_elapsed = toc;
  FFF_err = Ftilde-F_exact;
  FFF_rel_err_norm = max(abs(FFF_err(:)))/F_exact_norm;
  disp(sprintf('mbold_FFF = (%i,%i,%i)', mbold_FFF(1), mbold_FFF(2), mbold_FFF(3)))
  disp(sprintf('FFF error: %.2e', FFF_rel_err_norm))
  disp(sprintf('FFF elapsed time: %.2e\n', FFF_elapsed))
  summary_elapsed(1,idx) = HLF_elapsed;
  summary_elapsed(2,idx) = FFF_elapsed;
  summary_err(1,idx) = HLF_rel_err_norm;
  summary_err(2,idx) = FFF_rel_err_norm;
end
rmpath('../src')
figure
semilogy(summary_elapsed(1,:), summary_err(1,:), '-*g', ...
         summary_elapsed(2,:), summary_err(2,:), '-or')
legend('HLF method', 'FFF method')
xlabel('Wall-clock time')
ylabel('Achieved accuracy')
title(sprintf('HLF function decomposition (d = %d)', d))
grid on
drawnow
