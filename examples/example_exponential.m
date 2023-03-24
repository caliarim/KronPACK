% Example of linear evolutionary equation (see [CCZ23, Sec. 4.4])
%
% Equation:
% \partial_t u + \sum_\mu \beta_\mu \partial_{x_mu}(x_\mu u) = ...
% \alpha\sum_\mu \beta_\mu^2\partial_{x_\mu}(x_\mu^2\partial_{x_\mu}u) - \gamma u
% Spatial domain: [a1,b1]x[a2,b2]x[a3,b3]=[0,2]^3
% Time domain: [0,tstar] = [0,1/2]
% Boundary conditions:
% Homogeneous Dirichlet at x_\mu = 0, \mu = 1,2,3
% Homogeneous Neumann at x_\mu = 2, \mu = 1,2,3
% Space discretization: second order centered finite differences with
%                       nbold uniformely distributed nodes
% Time integration method: Exponential, ode23 and RK4
%
% [CCZ23] M. Caliari, F. Cassini, and F. Zivcovich,
%         A mu-mode BLAS approach for multidimensional tensor-structured
%         problems, NUMERICAL ALGORITHMS 92, 2483-2508 (2023)

clear all
addpath('../src')
fprintf('---- Linear evolutionary equation ----\n')
d = 3;
a = zeros(1, d);
b = 2*ones(1, d);
nbold = 5*(10:12);
alpha = 1/2;
beta = 2/3*ones(1, d);
gamma = 1/100;
tstar = 1/2;
ts_tucker = 1;
tau = tstar/ts_tucker;
for mu = 1:d
  x{mu} = linspace(a(mu), b(mu), nbold(mu)+1)';
  x{mu} = x{mu}(2:nbold(mu)+1);
  h(mu) = (b(mu)-a(mu))/nbold(mu);
  D1{mu} = spdiags(ones(nbold(mu), 1)*([-1, 1]/(2*h(mu))), [-1, 1], nbold(mu), nbold(mu));
  D1{mu}(nbold(mu), nbold(mu)-1:nbold(mu)) = [0, 0];
  D2{mu} = spdiags(ones(nbold(mu), 1)*([1, -2, 1]/(h(mu)^2)), -1:1, nbold(mu), nbold(mu));
  D2{mu}(nbold(mu), nbold(mu)-1:nbold(mu)) = [2, -2]/(h(mu)^2);
  A{mu} = spdiags(((2*alpha*beta(mu)-1)*beta(mu))*x{mu}, 0, nbold(mu), nbold(mu))*D1{mu} + ...
         spdiags((alpha*beta(mu)^2)*x{mu}.^2, 0, nbold(mu), nbold(mu))*D2{mu} + ...
         (-beta(mu)-gamma/3)*speye(nbold(mu));
  Afull{mu} = full(A{mu});
end
[X{1:d}] = ndgrid(x{1:d});
U0 = (X{1}-a(1)).*(b(1)-X{1}).^2;
for mu = 2:d
  U0 = U0.*(X{mu}-a(mu)).*(b(mu)-X{mu}).^2;
end
% tucker
fprintf('Tucker\n')
Utucker{1} = U0;
tic
for mu = 1:d
   E{mu} = expm(tau*Afull{mu});
end
for k = 1:ts_tucker
  Utucker{k+1} = tucker(Utucker{k}, E);
end
tucker_elapsed = toc;
fprintf('Time steps: %i\n', ts_tucker)
fprintf('Elapsed time: %.2e\n\n', tucker_elapsed)
Uref = Utucker{ts_tucker+1};
Uref_norm = max(abs(Uref(:)));
% ode23 - Vector formulation
fprintf('ode23 - Vector formulation\n')
M = kronsum(A);
odefun = @(t, y) M*y;
tic
[tode23_vec, Uode23_vec] = ode23(odefun, [0, tstar], U0(:));
ode23_vec_elapsed = toc;
Uode23_vec = reshape(Uode23_vec(end,:).', nbold);
ode23_vec_err = Uref-Uode23_vec;
ode23_vec_rel_err_norm = max(abs(ode23_vec_err(:)))/Uref_norm;
fprintf('Time steps: %i\n', length(tode23_vec))
fprintf('Elapsed time: %.2e\n', ode23_vec_elapsed)
fprintf('Error: %.2e\n\n', ode23_vec_rel_err_norm)
% ode23 - Tensor formulation
fprintf('ode23 - Tensor formulation\n')
odefun = @(t, y) reshape(kronsumv(reshape(y, nbold), Afull), [], 1);
tic
[tode23, Uode23] = ode23(odefun, [0, tstar], U0(:));
ode23_elapsed = toc;
Uode23 = reshape(Uode23(end,:).', nbold);
ode23_err = Uref-Uode23;
ode23_rel_err_norm = max(abs(ode23_err(:)))/Uref_norm;
fprintf('Time steps: %i\n', length(tode23))
fprintf('Elapsed time: %.2e\n', ode23_elapsed)
fprintf('Error: %.2e\n\n', ode23_rel_err_norm)
% RK4 - Vector formulation
fprintf('RK4 - Vector formulation\n')
ts_rk4 = 1351;
taurk = tstar/ts_rk4;
rk4_vec_fun = @(y) M*y;
Urk4_vec = U0(:);
tic
for k = 1:ts_rk4
  s1 = rk4_vec_fun(Urk4_vec);
  s2 = rk4_vec_fun(Urk4_vec+taurk/2*s1);
  s3 = rk4_vec_fun(Urk4_vec+taurk/2*s2);
  s4 = rk4_vec_fun(Urk4_vec+taurk*s3);
  Urk4_vec = Urk4_vec+taurk/6*(s1+2*s2+2*s3+s4);
end
rk4_vec_elapsed = toc;
Urk4_vec = reshape(Urk4_vec, nbold);
rk4_vec_err = Uref-Urk4_vec;
rk4_vec_rel_err_norm = max(abs(rk4_vec_err(:)))/Uref_norm;
fprintf('Time steps: %i\n', ts_rk4)
fprintf('Elapsed time: %.2e\n', rk4_vec_elapsed)
fprintf('Error: %.2e\n\n', rk4_vec_rel_err_norm)
% RK4 - Tensor formulation
fprintf('RK4 - Tensor formulation\n')
rk4_fun = @(y) kronsumv(y, Afull);
Urk4 = U0;
tic
for k = 1:ts_rk4
  s1 = rk4_fun(Urk4);
  s2 = rk4_fun(Urk4+taurk/2*s1);
  s3 = rk4_fun(Urk4+taurk/2*s2);
  s4 = rk4_fun(Urk4+taurk*s3);
  Urk4 = Urk4+taurk/6*(s1+2*s2+2*s3+s4);
end
rk4_elapsed = toc;
rk4_err = Uref-Urk4;
rk4_rel_err_norm = max(abs(rk4_err(:)))/Uref_norm;
fprintf('Time steps: %i\n', ts_rk4)
fprintf('Elapsed time: %.2e\n', rk4_elapsed)
fprintf('Error: %.2e\n\n', rk4_rel_err_norm)
rmpath('../src')
