function res = ifft2c(x)
if(~strcmp(class(x),'gpuArray')) %Don't do gpu computation
fctr = size(x,1)*size(x,2);
res = zeros(size(x));

size_x = size(x);

x = reshape(x, size_x(1), size_x(2), []);

for ii=1:size(x,3)
        res(:,:,ii) = sqrt(fctr)*fftshift(ifft2(ifftshift(x(:,:,ii))));
end

res = reshape(res, size_x);

else    %do gpu computation
    dims = size(x);
    fctr = dims(1)*dims(2);
    res  = reshape(sqrt(fctr)*fftshift(fftshift(ifft2(ifftshift(ifftshift(reshape(x,dims(1),dims(2),[]),1),2)),1),2),dims);
end

end
