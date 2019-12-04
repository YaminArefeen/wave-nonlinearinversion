function out = nlforwardoperator(x,ops,W)
%Performs the nonlinear forward operator for wave-encoded nonlinear inversion.  Currently assuming 2D-imaging and one set of coil sensitivity maps
%
%Inputs
%   x  - M x N x (C + 1),      Current estimate of the image and coil sensitivity maps
%   ops- Struct                Contains all the forward wave-SENSE operators. 
%   W  - Function Handle       Performs the transformation that penalizes high frequencies in coil maps during reconstruction
%Outputs
%   out- M x N x C             Estimate of acquired data

%Essentially perform the wave-SENSE operator
%1. Apply the coil sensitivities to the image:                  bsxfun(@times,x(:,:,1),W(x(:,:,2:end),'-f')
%2. Resize the result for wave related oversampling:            ops.Rsz
%3. Apply the fourier transform along the x dimension:          ops.Fx
%4. Apply the wave encoding pointspread function:               ops.Wave
%5. Apply the fourier transform along the y dimension:          ops.Fy
%6. Apply the undersampling mask:                               ops.R
%
%NOTE: We apply the forward W operator on the coil sensitivity maps to transform them back to the appropriate image space before applying the forward operator.

if(length(size(x)) == 3) %2D problem
    out = ops.R(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(bsxfun(@times,x(:,:,1),W(x(:,:,2:end),'-f')))))));
elseif(length(size(x)) == 4) %3D problem
    out = ops.R(ops.Fz(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(bsxfun(@times,x(:,:,:,1),W(x(:,:,:,2:end),'-f'))))))));
end

end
