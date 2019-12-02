function [psf,gradient] = wavepsf3d(adc,Tadc,gamma,Gymax,Gzmax,yind,zind,cycles)
nx = length(adc);
ny = length(yind);
nz = length(zind);

gradient = sin(cycles * pi * adc ./ Tadc);

psf = zeros(nx,ny,nz);
for y = 1:ny
    for z = 1:nz
        psf(:,y,z) = exp(sqrt(-1) .* gamma .* Tadc .* gradient .* (Gymax .* yind(y) + Gzmax .* zind(z)));
    end
end

end %end of function
