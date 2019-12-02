function out = nlforwardjacobian(dx,xn,ops,W)
%Computes the matrix-vector product between the Jacobian (evaluated at xn) of the nonlinear wave forward operator and the vector dx = (dm, dc1, dc2, ... dcN)
%
%Inputs
%   dx      - M x N x (C + 1)        Current estimate of the delta change in the image and coil sensitivities
%   ops     - struct                 Contains all the forward wave-SENSE operators
%   W       - Function Handle        Performs the transformation that penalizes high frequence in coil maps during reconstruction
%Outputs
%   out     - M x N x C              Result of applying the Jacobian to dx

%We want to compute the following:
%               (R * Fy * Wave * Fx * Rsz * (c1 * dm + m * dc1)
%                           .
% DF(xn) dx =               .
%                           .
%               (R * Fy * Wave * Fx * Rsz * (cN * dm + m * dcN)

%1. Compute ci * dm + m * dci for each coil.
%2. Apply the remainder of the forward WAVE operator
%   a. Resize the result of the wave related oversampling:      ops.Rsz
%   b. Apply the Fourier transform along the x dimension:       ops.Fx
%   c. Apply the wave encoding pointspread function:            ops.Wave
%   d. Apply the Fourier transform along the y dimension:       ops.Fy
%   e. Apply the undersampling mask:                            ops.R
%
%NOTE: We apply the forward W operator on the coil sensitivity maps to transform them back to the appropriate image space before applying the forward operator

if(length(size(dx)) == 3) %2D problem
    out = ops.R(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(bsxfun(@times,dx(:,:,1),W(xn(:,:,2:end),'-f')) + bsxfun(@times,xn(:,:,1),W(dx(:,:,2:end),'-f')))))));
elseif(length(size(dx)) == 4) %3D problem
    out = ops.R(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(bsxfun(@times,dx(:,:,:,1),W(xn(:,:,:,2:end),'-f')) + bsxfun(@times,xn(:,:,:,1),W(dx(:,:,:,2:end),'-f')))))));
end

end %End of function
