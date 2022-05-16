% Example of semilinear evolutionary equation (see [CCZ22, Sec. 4.5])
%
% Equation:
% \partial_t u(t,x) = \Delta u(t,x) + 1/(1+u(t,x)^2) + \Phi(t,x)
% Spatial domain: [a1,b1]x[a2,b2]x[a3,b3]=[0,1]^3
% Time domain: [0,tstar] = [0,1]
% Boundary conditions:
% Homogeneous Dirichlet in all directions
% Space discretization: second order centered finite differences with
%                       nbold uniformely distributed nodes
% Time integration method: Backward-Forward Euler
%
% [CCZ22] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional
%         tensor-structured problems, Submitted 2022

clear all
addpath('../src')
disp(sprintf('---- Semilinear evolutionary equation ----'))
d = 3;
a = zeros(1, d);
b = ones(1, d);
nbold = 4*[10:12];
tstar = 1;
ts = 100;
tau = tstar/ts;
for mu = 1:d
  x{mu} = linspace(a(mu), b(mu), nbold(mu)+2).';
  x{mu} = x{mu}(2:nbold(mu)+1);
  h(mu) = (b(mu)-a(mu))/(nbold(mu)+1);
  A{mu} = spdiags(ones(nbold(mu), 1)*([1, -2, 1]/(h(mu)^2)), -1:1, nbold(mu), nbold(mu));
  M{mu} = 1/d*speye(nbold(mu)) - tau*A{mu};
  Mfull{mu} = full(M{mu});
  P{mu} = eye(nbold(mu)) - full(tau*A{mu});
end
[X{1:d}] = ndgrid(x{:});
u0 = (X{1}-a(1)).*(b(1)-X{1});
for mu = 2:d
  u0 = u0.*(X{mu}-a(mu)).*(b(mu)-X{mu});
end
u0 = u0(:);

x_sp = 0;
for mu = 1:d
  tmp = 1;
  for mu2 = [1:mu-1,mu+1:d]
    tmp = tmp.*(X{mu2}-a(mu2)).*(b(mu2)-X{mu2});
  end
  x_sp = x_sp+tmp;
end
x_sp = x_sp(:);
f = @(t, u) 1./(1+u.^2)+exp(t)*u0+2*exp(t)*x_sp-1./(1+(exp(t)*u0).^2);
u_exact = exp(tstar)*u0;
u_exact_norm = max(abs(u_exact));
M = kronsum(M); % cell overwritten with matrix
% Direct method
disp(sprintf('Direct method'))
[R, ~, Pmat] = chol(M);
Rt = R';
uk = Pmat'*u0;
u0_p = Pmat'*u0;
x_sp_p = Pmat'*x_sp;
fchol = @(t, u) 1./(1+u.^2)+exp(t)*u0_p+2*exp(t)*x_sp_p-1./(1+(exp(t)*u0_p).^2);
tic
for i = 1:ts
  tk = tau*(i-1);
  uk = R\(Rt\(uk+tau*fchol(tk, uk)));
end
direct_elapsed = toc;
uk = Pmat*uk;
direct_err = uk-u_exact;
direct_rel_err_norm = max(abs(direct_err))/u_exact_norm;
disp(sprintf('Error: %.2e', direct_rel_err_norm))
disp(sprintf('Elapsed time: %.2e\n', direct_elapsed))
% CG (vector)
disp(sprintf('CG method - vector'))
maxit = 100;
tol = min(h)^2/10;
uk = u0;
tic
for i = 1:ts
  tk = tau*(i-1);
  [uk, ~, ~, iter(i)] = pcg(M, uk+tau*f(tk,uk), tol, maxit, [], [], uk);
end
CG_vec_elapsed = toc;
CG_vec_err = uk-u_exact;
CG_vec_rel_err_norm = max(abs(CG_vec_err))/u_exact_norm;
disp(sprintf('Avg. iterations per time step: %i', ceil(mean(iter))))
disp(sprintf('Error: %.2e', CG_vec_rel_err_norm))
disp(sprintf('Elapsed time: %.2e\n', CG_vec_elapsed))
% CG (tensor)
disp(sprintf('CG method - tensor'))
uk = u0;
Mfun = @(x) reshape(kronsumv(reshape(x, nbold), Mfull), [], 1);
tic
for i = 1:ts
  tk = tau*(i-1);
  [uk, ~, ~, iter(i)] = pcg(Mfun, uk+tau*f(tk,uk), tol, maxit, [], [], uk);
end
CG_elapsed = toc;
CG_err = uk-u_exact;
CG_rel_err_norm = max(abs(CG_err))/u_exact_norm;
disp(sprintf('Avg. iterations per time step: %i', ceil(mean(iter))))
disp(sprintf('Error: %.2e', CG_rel_err_norm))
disp(sprintf('Elapsed time: %.2e\n', CG_elapsed))
% PCG
disp(sprintf('PCG method'))
uk = u0;
Mfun = @(x) reshape(kronsumv(reshape(x, nbold), Mfull), [], 1);
Pfun = @(x) reshape(itucker(reshape(x, nbold), P), [] ,1);
tic
for i = 1:ts
  tk = tau*(i-1);
  [uk, ~, ~, iter(i)] = pcg(Mfun, uk+tau*f(tk,uk), tol, maxit, Pfun, [], uk);
end
PCG_elapsed = toc;
PCG_err = uk-u_exact;
PCG_rel_err_norm = max(abs(PCG_err))/u_exact_norm;
disp(sprintf('Avg. iterations per time step: %i', ceil(mean(iter))))
disp(sprintf('Error: %.2e', PCG_rel_err_norm))
disp(sprintf('Elapsed time: %.2e\n', PCG_elapsed))
rmpath('../src')
