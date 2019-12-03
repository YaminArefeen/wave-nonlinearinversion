function hist = regnewton(xo,p,ops)
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

fprintf('Starting Newton Algorithm...\n')
fprintf('Newton Iterations         : %d\n',it)
fprintf('L2 on maps and parameters : %f\n',ao)
fprintf('Regularization scaling    : %.3f\n',q)

for n = 1:it    %Do the newton iterations
    fprintf('~~~~~~~~~Newton Iteration: %d~~~~~~~~~\n',n)
%At each iteration, should be solving the l2-regularized linearized system:
%       (DF(xn)^H * DF(xn) + I an) * dx = DF(xn)^H * (y - F(xn)) - an (xn - xo)
    fprintf('Loading y... ')
    y = readcfl('/home/yarefeen/research/wave-nonlinearinversion/results/curiter/y'); %Reload y
    fprintf('done\n')
    
    an           = ao * q^(n-1);             %Regularization level at this iteration

    fprintf('Computing forward operator... ')
    Fxn          = ops.F(xn);                %Current attempt to match the acquired data
    fprintf('done\n')
    
    fprintf('Current Netwon Residual: ') 
    res(n)       = norm(y(:) - Fxn(:))/norm(y(:));
    fprintf('%.4f\n',res(n))

    tmp     = y - Fxn;
    clear Fxn

    fprintf('Computing right hand side for linear system... ')
    rhs     = ops.DFH(tmp,xn) - an * (xn - xo);        %Right hand side in the linearized system
    fprintf('done\n')
    clear tmp y

    A       = @(dx) reshape(ops.DFH(ops.DF(reshape(dx,dims),xn),xn),numel(xn),1) + an * dx;  
    fprintf('Solving linearized system with pcg...\n')
    dx      = reshape(pcg(A,rhs(:)),dims);              %Solve the system with Conjuage-gradients

    clear rhs %Clear for the sake of memory

    xn      = xn + dx;                                  %Perform the Newton Update

    clear dx %Clear for the sake of memory
end

Fxn               = ops.F(xn);
res(n+1)          = norm(y(:) - Fxn(:))/norm(y(:));

hist.xest   = xn;
hist.res    = res;
end
