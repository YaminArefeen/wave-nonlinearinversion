function [psf,gradients] = wavepsf3d(adc,Tadc,gamma,Gymax,Gzmax,yind,zind,cycles)
nx = length(adc);
ny = length(yind);
nz = length(zind);

gradienty = sin(cycles * pi * adc ./ Tadc);
gradientz = cos(cycles * pi * adc ./ Tadc);
%In 3D, the gradient wave forms are 90 degrees off phase

psf = zeros(nx,ny,nz);
for y = 1:ny
    for z = 1:nz
        psf(:,y,z) = exp(sqrt(-1) .* gamma .* Tadc .* (Gymax .* gradienty.* yind(y) + Gzmax .* gradientz .* zind(z)));
    end
end

gradients = [gradienty;gradientz]; %Store gradients for visualization

end %end of function
