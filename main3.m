%%A first pass at a 3D implementation of nonlinear inversion with WAVE encoding
addpath('util')

fprintf('Loading ground truth image and coil sensitivity maps... ')
if(~exist('M'))
gt   = readcfl('phantom/gt3d');
maps = readcfl('phantom/maps3d');
[M,N,P,C] = size(maps);
fprintf('done\n')
else
    fprintf('data already loaded\n')
end

fprintf('Defining constants and flags... ')
%-Acceleration, acs size, and noise level parameters
seflag  = 0;          %Whether we want to perform SENSE reconstruction before nonlinear inversion
R       = [2,2];      %Undersampling factor in the phase encode and partition directions
acss    = [24,24];    %Acs size in the phase encode and partition directions
stdn    = 0;          %Noise level
os      = 6;          %Wave oversampling 

Nro     = M*os;
acspe   = (N/2 - (acss/2 - 1)):(N/2 + acss/2); %Phase encode acs size
acspa   = (P/2 - (acss/2 - 1)):(P/2 + acss/2); %partition phase encode size

%-Wave Parameters
wvflag      = 0;
cycles      = 8;
FOVy        = 160 * 1e-3;
FOVz        = 160 * 1e-3;
Tadc        = 1432.7 * 1e-6;
Gymax       = 16*1e-3;
Gzmax       = 16*1e-3;
gamma       = 42.57747892 * 1e6;
yind        = linspace(-FOVy/2,FOVy/2,N);
zind        = linspace(-FOVz/2,FOVz/2,P);
adc         = linspace(0,Tadc,Nro);

%-Nonlinear inversion parameters
gpu     = 0;
l       = 32;
s       = 300;
p.it    = 15;
p.ao    = .01;
p.q     = 2/3;

xx = repmat(reshape(linspace(-.5,.5,M),M,1,1),1,N,P);
yy = repmat(reshape(linspace(-.5,.5,N),1,N,1),M,1,P);
zz = repmat(reshape(linspace(-.5,.5,P),1,1,P),M,N,1);
k  = xx.^2 + yy.^2 + zz.^2;
No = (1 + s * k) .^ (l/2);
fprintf('done\n')

fprintf('Generating wave psf, linear ops, and data... ')
if(wvflag)  
    [psf,gradient] = wavepsf3d(adc,Tadc,gamma,Gymax,Gzmax,yind,zind,cycles);
else
    psf = 1;
    os  = 1;
    Nro = M;
end

%-Defining our undersampling mask
mask = zeros(Nro,N,P,C);
mask(:,1:R(1):end,1:R(2):end,:) = 1;
mask(:,acspe,acspa,:)           = 1;

if(gpu)
    mask = gpuArray(mask);
    psf  = gpuArray(psf);
    maps = gpuArray(maps);
    %No   = gpuArray(No);
    gt   = gpuArray(gt);
end
%-Define the wave sense operators
ops = definewavesenseops3d(maps,mask,psf,os);

%-Acquired data
y   = ops.A_for(gt);
y   = y / norm(y(:)) * 100;
%NOT ADDING NOISE FOR NOISE SINCE I ASSUME IT EXISTS IN ACQUIRED DATA
fprintf('done\n')

%Perform a SENSE Reconstruction to compare to nonlinear inversion 
if(seflag)
    fprintf('SENSE reconstruction...\n')
    kadj    = ops.A_adj(y);
    clear   y;  %for memory issues
    tic
    sense   = reshape(pcg(ops.pcgA,kadj(:)),M,N,P); 
    toc
    writecfl('results/sensenowave/experiment1/res',sense)
    return
end

fprintf('Defining nonlinear operators... ')
nl.W    = @(c,type) coilprecon3d(c,No,type);
nl.F    = @(x)      nlforwardoperator(x,ops,nl.W);
nl.DF   = @(dx,xn)  nlforwardjacobian(dx,xn,ops,nl.W);
nl.DFH  = @(dy,xn)  nladjointjacobian(dy,xn,ops,nl.W);
fprintf('done\n')

fprintf('Computing initial guess as zerofilled reconstruction... ')
xo = zeros(M,N,P,C+1);
xo(:,:,:,1) = ops.A_adj(y);
fprintf('done\n')

fprintf('Writing data for memory... ')
writecfl('results/curiter/y',y) 
clear y %for the sake of memory management
fprintf('done\n')

tic
hist = regnewton3d(xo,p,nl);
toc

fprintf('Saving estimated proton density and coil maps... ')
cest = nl.W(hist.xest(:,:,:,2:end),'-f');
cnorm= sqrt(sum(abs(cest).^2,4));

pest = hist.xest(:,:,:,1) .* cnorm;
cest2= bsxfun(@rdivide,cest,cnorm);
clear cest hist cnorm

writecfl('results/nlnowave/experiment1/pest',pest)
clear pest
writecfl('results/nlnowave/experiment1/cest',cest2)
clear cest2
fprintf('done\n')
