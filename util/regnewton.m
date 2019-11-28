function hist = regnewton(xo,p,y,ops)
%Performs a l2 regularized newton solve. Each linearized system is solved with the conjugate gradient algorithm and L2-regularization
%Inputs:
    %xo     - Initial guess at a solution
    %p      - Parameters for the optimization (step size, initial reg, scaling of reg per iteration)
    %y      - Acquired data
    %ops    - Should contain the forward operator, Jacobian Operator, and Adjoint Jacobian Operator
%Outputs:
    %hist   - Contains solution and residual history over time

ao      = p.ao;            %Initial l2 regularization parameter
q       = p.q;             %How much to scale the regularization parameter at each step
it      = p.it;            %Number of newton iterations

dims    = size(xo);        %Dimensions of the parameters so that I can do reshaping for matlab's pcg
xn      = xo;              %Set initial guess

res     = zeros(it+1,1);            %Store the residuals at each iteration (and beginning)
sol     = zeros([size(xn) it+1]);   %Store the solution at each iteration

%Store everything onto the gpu if I have chosen that option
if(strcmp(class(y),'gpuArray'))
    res = gpuArray(res);
    sol = gpuArray(sol);
end

fprintf('Starting Newton Algorithm...\n')
fprintf('Newton Iterations         : %d\n',it)
fprintf('L2 on maps and parameters : %f\n',ao)
fprintf('Regularization scaling    : %.3f\n',q)

for n = 1:it    %Do the newton iterations
    fprintf('~~~~~~~~~Newton Iteration: %d~~~~~~~~~\n',n)
%At each iteration, should be solving the l2-regularized linearized system:
%       (DF(xn)^H * DF(xn) + I an) * dx = DF(xn)^H * (y - F(xn)) - an (xn - xo)

    an           = ao * q^(n-1);             %Regularization level at this iteration

    Fxn          = ops.F(xn);                %Current attempt to match the acquired data

    res(n)       = norm(y(:) - Fxn(:))/norm(y(:));
    sol(:,:,:,n) = xn;                       %Computing residual and storing current solution

    fprintf('Current Netwon Residual: %.4f\n',res(n))

    rhs     = ops.DFH(y-Fxn,xn) - an * (xn - xo);        %Right hand side in the linearized system
    A       = @(dx) reshape(ops.DFH(ops.DF(reshape(dx,dims),xn),xn),numel(xn),1) + an * dx;  
    dx      = reshape(pcg(A,rhs(:)),dims);              %Solve the system with Conjuage-gradients

    clear Fxn; %Clear for the sake of memory

    xn      = xn + dx;                                  %Perform the Newton Update
end

Fxn               = ops.F(xn);
res(n+1)          = norm(y(:) - Fxn(:))/norm(y(:));
sol(:,:,:,n+1)    = xn;

hist.xest   = xn;
hist.res    = res;
hist.sol    = sol;
end
