function [psf,gradient] = wavepsf(adc,Tadc,gamma,Gymax,yind,cycles)
%In this function, we generate a wave point spread function and gradient given the relevant wave parameters
%
%Inputs:
%adc (Nro x 1) - Times at which ADC will be sampled during the read out.  Note, Nro corresponds to the number of readout points, not the image dimension pixels
%Tadc          - Total read out time
%gamma         - Larmor Frequency at Hz/T
%Gymax         - Maximum gradient amplitude T/m
%yind (M x1)   - Physical y location of phase encode points 
%cycles        - Number of gradient cycles

%Outputs:
%psf (Nro x M)      - Wave point spread function
%gradient (Nro x 1) - Integral of simulated gradient played out during acquisition

nx = length(adc);
ny = length(yind);

gradient = sin(cycles*pi*adc ./ Tadc);                  % integral of cosine wave gradient -> sine

psf = zeros(nx,ny);
for y = 1:ny
    psf(:,y) = exp(sqrt(-1) .* gamma .* Tadc .* Gymax .* gradient .* yind(y));
end


end %end of function 
