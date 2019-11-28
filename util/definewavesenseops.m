function ops = definewavesenseops(maps,mask,psf,sc)
%This function defines all of the relevant linear operators and constructs a wave-encoding forward operator for the simple regime of 2D-imaging and one set of coil sensitivity maps.
%
%Inputs:
%   maps - M x N x C        Pre-determined, single set of coil sensitivity maps
%   mask - M x N x C        Undersampling mask
%   psf  - Nro x N x C      Wave point spread function used to capture the effect of sinusoidal gradients:w
%   sc   - scalar           Readout oversampling factor

[M,N,C] = size(maps);

ops.S_for = @(x) bsxfun(@times,maps,permute(x,[1,2,4,3]));              %M x N -> M x N x C
ops.S_adj = @(Sx) squeeze(sum(bsxfun(@times,conj(maps),Sx),3));         %M x N x C -> M x N
%Forward and adjoint coil sensitivity operators

ops.F_for = @(x) mfft2(x);      %Forward 2-D fourier transform
ops.F_adj = @(x) mifft2(x);     %Adjoint 2-D fourier transform
ops.Fx    = @(x) mfft(x,1);     %Forward 1-D Fourier transform along x 
ops.Fy    = @(x) mfft(x,2);     %Forward 1-D Fourier transform along y
ops.Fxadj = @(x) mifft(x,1);    %Adjoint 1-D Fourier transform along x
ops.Fyadj = @(x) mifft(x,2);    %Adjoing 1-D Fourier transform along y

ops.R = @(x) bsxfun(@times,mask,x); %undersampling mask operator

ops.Wave     = @(x) bsxfun(@times,x,psf); %forward wave operator
ops.Wave_adj = @(x) bsxfun(@times,x,conj(psf)); %adjoit wave operator

ops.Rsz    = @(x) resizewave(x,M,sc);
ops.Rszadj = @(x) cropwave(x,N);
%resizing operators for wave.

ops.A_for = @(x) ops.R(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(ops.S_for(x))))));        %Full wave forward model
ops.A_adj = @(x) ops.S_adj(ops.Rszadj(ops.Fxadj(ops.Wave_adj(ops.Fyadj(x)))));  %Full wave adjoint model

ops.AhA = @(x) ops.S_adj(ops.Rszadj(ops.Fxadj(ops.Wave_adj(ops.Fyadj(ops.R(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(ops.S_for(x)))))))))));
%Forward and adjoint wave SENSE operator

ops.pcgA        = @(x) vec(ops.AhA(reshape(x,M,N)));  %Function handle for matlab's pcg

ops.analyzepsf  = @(x) mifft(squeeze(psf(:,:,1).*mfft(ops.Rsz(x),1)),1);
%analyze effect of psf on image space with no undersampling or coiils

end

function out = resizewave(in,M,sc)
    dim = size(in);    
    if(length(dim) == 3)
        out = zeros(sc*M,dim(2),dim(3));
        if(strcmp(class(in),'gpuArray'))
            out = gpuArray(out);
        end
        out(end/2-(M/2-1):end/2+M/2,:,:) = in;
    elseif(length(dim) == 4)
        out = zeros(sc*M,dim(2),dim(3),dim(4));
        if(strcmp(class(in),'gpuArray'))
            out = gpuArray(out);
        end
        out(end/2-(M/2-1):end/2+M/2,:,:,:) = in;
    elseif(length(dim) == 2)
        out = zeros(sc*M,dim(2));
        if(strcmp(class(in),'gpuArray'))
            out = gpuArray(out);
        end
        out(end/2-(M/2-1):end/2+M/2,:) = in;
    end
end

function out = cropwave(in,M)
    dim = size(in);
    if(length(dim) == 3)
        out = in(end/2-(M/2-1):end/2+M/2,:,:);
    elseif(length(dim) == 4)
        out = in(end/2-(M/2-1):end/2+M/2,:,:,:);
    elseif(length(dim) == 2)
        out = in(end/2-(M/2-1):end/2+M/2,:);
    end
end
