% Code validation for TUCKER function of KronPACK (see [CCZ22, Sec. 4.1])
%
% To execute this script, it is required to have in path the toolboxes
% Tensor Toolbox for MATLAB and Tensorlab.
% Moreover, as Tensor Toolbox for MATLAB does not have GNU Octave support,
% this script can't be exectuted in GNU Octave.
%
% [CCZ22] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional
%         tensor-structured problems, Submitted 2022

clear all
cv = 1;
try
  tmprod(rand(2),{rand(2),rand(2)},1:2);
catch
  disp('Tensorlab not found in path.')
  cv = 0;
end
try
  ttm(tensor(rand(2)),{rand(2),rand(2)});
catch
  disp('Tensor Toolbox for MATLAB not found in path.')
  cv = 0;
end

if cv
  addpath('../src')
  disp(sprintf('---- Code validation ----'))
  drange = [3*ones(1,4),6*ones(1,4)];
  nrange = [(12:2:18).^2,(12:2:18)];
  nrepsrange = [100,50,20,10,100,50,20,10];
  for j = 1:length(drange)
    d = drange(j);
    n = nrange(j);
    nreps = nrepsrange(j);
    nbold = n*ones(1,d);
    disp(sprintf('d = %i, n = %i',d,n))
    T = randn(nbold);
    Tconv = tensor(T);
    for mu = 1:d
      M{mu} = randn(nbold(mu));
    end
    % KronPACK
    tic
    for i = 1:nreps
      S_kp = tucker(T,M);
    end
    time_kp(j) = toc/nreps;
    % Tensorlab
    tic
    for i = 1:nreps
      S_tl = tmprod(T,M,1:d);
    end
    time_tl(j) = toc/nreps;
    % Tensor Toolbox for MATLAB
    tic
    for i = 1:nreps
      S_tt = ttm(Tconv,M);
    end
    time_tt(j) = toc/nreps;
    disp(sprintf('Elapsed time KronPACK: %.2e',time_kp(j)))
    disp(sprintf('Elapsed time Tensorlab: %.2e',time_tl(j)))
    disp(sprintf('Elapsed time Tensor Toolbox for MATLAB: %.2e',time_tt(j)))
    disp(sprintf('Error w/Tensorlab: %.2e',norm(S_tl(:)-S_kp(:),inf)/norm(S_kp(:),inf)))
    disp(sprintf('Error w/Tensor Toolbox for MATLAB: %.2e',norm(S_tt(:)-S_kp(:),inf)/norm(S_kp(:),inf)))
    disp(' ')
  end
  figure
  subplot(1,2,1)
  semilogy(nrange(1:4),time_tt(1:4),'xb-',...
           nrange(1:4),time_tl(1:4),'or-',...
           nrange(1:4),time_kp(1:4),'dg-')
  xticks(nrange(1:4))
  xlim([100,370])
  xlabel('n')
  ylabel('Wall-clock time (s)')
  title('Code validation (d = 3)')
  subplot(1,2,2)
  semilogy(nrange(5:8),time_tt(5:8),'xb-',...
           nrange(5:8),time_tl(5:8),'or-',...
           nrange(5:8),time_kp(5:8),'dg-')
  xticks(nrange(5:8))
  xlim([10,20])
  xlabel('n')
  title('Code validation (d = 6)')
  legend('ttm','tmprod','tucker',...
         'Location','SouthEast')
  drawnow
  rmpath('../src')
else
  warning('Code validation not performed.')
end
