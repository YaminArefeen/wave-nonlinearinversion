function out = coilprecon(c,No,type)
%Applies the coil preconditioning matrix which is used for coil regularization in the nonlinear inversion problem. In essence, want to apply the following to each coil in the forward problem
%   outi = (1 + s ||k||_2^2)^(l/2) F ci where N = (1 + s ||k||_2^2)^(l/2)
%Inputs
%   c   - M x N x P x C         Coils that we want to apply the preconditioning too
%   type- str                   Str indicating whether we want forward, inverse or adjoint
%   n   - M x N                 1 + s ||k||_2^2) ^(l/2)
%Outputs
%   out - M x N x P x C         Appropriately preconditioned coils

if(strcmp('-f',type))   %Perform the forward preconditioning
    out = ifft2c(bsxfun(@times,c,No.^(-1)));
    return
elseif(strcmp('-h',type)) %Perform the adjoint preconditioing operation
    out = bsxfun(@times,conj(No.^(-1)),fft2c(c));
    return
elseif(strcmp('-i',type)) %Perform the inverse preconditioning operation
    out = bsxfun(@times,No,fft2c(c));
    return
end

error('Correct type not written in preconditioning operation')
end
