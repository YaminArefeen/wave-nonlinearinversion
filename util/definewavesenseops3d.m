function ops = definewavesenseops3d(maps,mask,psf,sc)

[M,N,P,C] = size(maps);

ops.S_for = @(x)  bsxfun(@times,maps,x);                %M x N x P -> M x N x P x C
ops.S_adj = @(x)  sum(bsxfun(@times,conj(maps),x),4);   %M x N x P x C -> M x N x P
ops.F_for = @(x) mfft3(x);      %Forward 3-D Fourier Transform
ops.F_adj = @(x) mifft3(x);     %Adjoint 3-D Fourier Transform
ops.Fx    = @(x) mfft(x,1);
ops.Fy    = @(x) mfft(x,2);
ops.Fz    = @(x) mfft(x,3);
ops.Fxadj = @(x) mifft(x,1);
ops.Fyadj = @(x) mifft(x,2);
ops.Fzadj = @(x) mifft(x,3);    %Fourward and adjoint fourier transforms along the specified dimension

ops.R     = @(x) bsxfun(@times,mask,x); %Undersampling mask opertaor

ops.Wave    = @(x) bsxfun(@times,x,psf);
ops.Wave_adj= @(x) bsxfun(@times,x,conj(psf));

ops.Rsz     = @(x) resizewave(x,M,sc);
ops.Rszadj  = @(x) cropwave(x,N);

ops.A_for   = @(x) ops.R(ops.Fz(ops.Fy(ops.Wave(ops.Fx(ops.Rsz(ops.S_for(x)))))));
ops.A_adj   = @(x) ops.S_adj(ops.Rszadj(ops.Fxadj(ops.Wave_adj(ops.Fyadj(ops.Fzadj(x))))));

ops.AhA     = @(x) ops.A_adj(ops.A_for(x));
ops.pcgA    = @(x) vec(ops.AhA(reshape(x,M,N,P))); %Function handle for matlab's PCG

ops.analyzepsf = @(x) ops.Fxadj(psf(:,:,:,1).*ops.Fx(ops.Rsz(x)));
end%End of function

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
