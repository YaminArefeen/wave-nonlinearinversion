function out = adjointjacobian(dy,xn,ops,W)
%Performs the the adjoint of the jacobian of the nonlienar forwaard operator for wave-encoded nonlienar inversion.  Currently assuming 2D-imaging and one set of coil sensitivity maps.
%
%Inputs
%   dy - M x N x C          Output from the forward Jacobian Operator
%   ops- Struct             Contains all the adjoint wave-SENSE operators
%   W  - Function Handle    Performs the transformation that penalizes high frequecies in the coil maps during reconstruction
%Outputs
%   out- M x N x (C + 1)    Output of applying the adjoint jacobian operator to dy
%
%We want to compute the following
%                   (sum[conj(c_i).* (Rsz^H * Fx^H * Wave^H * Fy^H * R^H(dyi)),i])
%                   (conj(m_1).* (Rsz^H * Fx^H * Wave^H * Fy^H * R^H(dy1))       )
% DF^H(xn) dy =                         .
%                                       .
%                   (conj(m_N).* (Rsz^H * Fx^H * Wave^H * Fy^H * R^H(dyN))       )

%1. Pre compute tmp = (Rsz^H * Fx^H * Wave^H * Fy^H * R^H(dyi) for each i by applying the following adjoint wave operators to each dy_i
%   a.  Adjoint of the undersampling mask:              ops.R
%   b.  Adjoint of Fourier Transform along y:           ops.Fyadj
%   c.  Adjoint of the forward wave operator:           ops.Wave_adj
%   d.  Adjoint of the Fourier Transform along x:       ops.Fxadj
%   e.  Adjoint of the resize operator (crop):          ops.Rszadj
%2. Perform the adjoint coil sensitity operator from the curresnt estimate of the coil maps:
%   squeeze(sum(bsxfun(@times,conj(W(xn(:,:,2:end),'-f')),tmp),3));
%3. Compute conj(m) .* tmp for each coil profile

out = zeros(size(xn));
if(strcmp(class(dy),'gpuArray'))
    out = gpuArray(out);
end


if(length(size(dy)) == 3) %2D problem
    tmp = ops.Rszadj(ops.Fxadj(ops.Wave_adj(ops.Fyadj(ops.R(dy)))));

    out(:,:,1)     = squeeze(sum(bsxfun(@times,conj(W(xn(:,:,2:end),'-f')),tmp),3));
    out(:,:,2:end) = W(sum(bsxfun(@times,conj(xn(:,:,1)),tmp),4),'-h');
elseif(length(size(dy)) == 4) %3D problem
    tmp = ops.Rszadj(ops.Fxadj(ops.Wave_adj(ops.Fyadj(ops.Fzadj(ops.R(dy))))));

    out(:,:,:,1)     = sum(bsxfun(@times,conj(W(xn(:,:,:,2:end),'-f')),tmp),4);
    out(:,:,:,2:end) = W(bsxfun(@times,conj(xn(:,:,:,1)),tmp),'-h');
end

end %End of function
