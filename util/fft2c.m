function res = fft2c(x)
if(~strcmp(class(x),'gpuArray')) %Don't do computation on the gpu
    fctr = size(x,1)*size(x,2);
    res = zeros(size(x));

    size_x = size(x);

    x = reshape(x, size_x(1), size_x(2), []);

    % tic
    for ii=1:size(x,3)
        res(:,:,ii) = 1/sqrt(fctr)*fftshift(fft2(ifftshift(x(:,:,ii))));
    end
    % toc

    res = reshape(res, size_x);
else %Do the computation on the gpu
    dims = size(x);
    fctr = dims(1) * dims(2);

    res  = reshape(1/sqrt(fctr)*fftshift(fftshift(fft2(ifftshift(ifftshift(reshape(x, dims(1), dims(2), []),1),2)),1),2),dims);

end





