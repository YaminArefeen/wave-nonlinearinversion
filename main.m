clear; clc;
addpath('util')

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Loading the digital phantom data
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Loading data and constants... ')
load('phantom/gt.mat')      %Loading the ground truth image
load('phantom/maps.mat')    %Loading pre-saved coil sensitivity maps 
[M,N,C] = size(maps);
fprintf('done\n')

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Defining constants and flags
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Defining Constants and flags... ')
%-Simulating acquired data constants
R           = 4;                %Undersampling factor for the simulation experiment
acss        = 24;               %Callibration size
stdn        = 5e-3;             %Noise Level
os          = 6;                %Oversampling in readout direction

Nro         = M*os;             %Number of readout points defined by level of oversampling in the readout direction
acs         = (N/2-(acss/2 - 1)):(N/2+acss/2); %Acs region

%-Wave Parameters
wvflag  = 1;              %       Flag indicating whether we want to use Wave or not
cycles  = 8;              %       Number of gradient cycles
FOVy    = 300* 1e-3;      % (m)   Simulated FOV in the y direction used to characterize wave in y direction
Tadc    = 1432.7* 1e-6;   % (sec) Simulated acquisition time to characterize wave in y direction
Gymax   = 16* 1e-3;     % (T/m) Maximum wave gradient to chracterize wave in y direction
gamma   = 42.57747892 * 1e6;% (Hz/T) Larmor frequency to characterize wave gradient 
yind    = linspace(-FOVy/2,FOVy/2,N) ; % (m) Location of simulated y samples
adc     = linspace(0,Tadc,Nro);        % (sec) Simulated ADC times

%-Nonlinear Inversion Parameters
gpu   = 1;                %Whether we want to use a GPU through Matlab's interface
l     = 32;               %Parameter to penalize high frequencies in the coil sensitivity estimates 
s     = 300;              %Parameter to penalize high frequencies in the coil sensitivity estimates 
p.it  = 15;               %Newton iterations
p.ao  = .01;              %Initial L2-regularization value
p.q   = 2/3;              %How much we will scale ao by at each iteration of the newton algorithm

xx = repmat(linspace(-.5,.5,M),N,1);
yy = repmat(linspace(-.5,.5,N)',1,M);
k  = abs(xx + 1i*yy).^2;
No = (1 + s *k) .^ (l/2);
%To penalize the coil sensitivity maps, we will apply the following operator:
%                   No .* fft2c(coils)
%s and l determine the way in which high fequencies are penalized in coil sensitivity estimates   
fprintf('done \n')

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Wave psf, sampling mask, sense ops, and data
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('Generating wave psf, linear ops, and data... ')

if(wvflag)
%-Defining the wave pointspread function
[psf,gradient] = wavepsf(adc,Tadc,gamma,Gymax,yind,cycles);
psf = repmat(psf,1,1,C);
else
psf = 1;
end

%-Defining our undersampling mask
mask = zeros(Nro,N,C);
mask(:,1:R:end,:) = 1;
mask(:,acs,:)     = 1;

%Transfer all relevant variables to GPU through Matlab's interface if gpu option chosen
if(gpu)
    mask = gpuArray(mask);
    psf  = gpuArray(psf);
    maps = gpuArray(maps);
    No   = gpuArray(No);
    gt   = gpuArray(gt);
end

%Define the WAVE sense operators
ops = definewavesenseops(maps,mask,psf,os);

%-Acquired data
y           = ops.A_for(gt);        %Noiseless undersampled data generated through the Wave-forward operator
y           = y / norm(y(:)) * 100; %Normalized to have norm 100
y           = addnoise(y,stdn,mask);%Adding noise to the data at the sampled points
fprintf('done\n')


%Perform a SENSE reconstruction to compare to the nonlinear inversion solution
fprintf('SENSE reconstruction...\n')
kadj    = ops.A_adj(y);         
sense   = reshape(pcg(ops.pcgA,kadj(:)),M,N);

fprintf('Defining nonlinear operators... ')
nl.W   = @(c,type) coilprecon(c,No,type);               %Operator to penalize high frequencies in coil
nl.F   = @(x)      nlforwardoperator(x,ops,nl.W);       %Nonlinear wave-encoding forward operator
nl.DF  = @(dx,xn)  nlforwardjacobian(dx,xn,ops,nl.W);   %Jacobian of the nonlinear wave-encoding forward operator
nl.DFH = @(dy,xn)  nladjointjacobian(dy,xn,ops,nl.W);   %Adjoint of the jacobian of the nonlinear-wave encoding forward operator
fprintf('done\n')

%Initial guess at a solution for the nonlinear problem.  Set the guess of proton density to 1 and guess of all initial coil maps to 0
xo = zeros(M,N,C+1);
xo(:,:,1) = 1;

%Perform a a regularized Newton reconstruction to jointly solve for the proton density and coil sensitivity maps
hist = regnewton(xo,p,y,nl);

cest = gather(nl.W(hist.xest(:,:,2:end),'-f'));         %Estimated coil sensitivity maps without scaling correction
cnorm= sqrt(sum(abs(cest).^2,3));                       %Scaling correction factor (don't know why this works yet, just took it straight from Martin's paper)

pest = gather(hist.xest(:,:,1) .* cnorm);               %Scaling correction on proton density estimates
cest2= cest./cnorm;                                     %Scaling correction on coil sensitivity estimates

a = @(x) gather(abs(x) / max(abs(x(:))));
out = [a(sense) a(pest) a(gt)];                         %Compare the reconstructions

results.out = out;
results.psf = gather(psf(:,:,1));
results.cest= cest2;
results.maps= gather(maps);
save('~/out.mat','results')                                %Save the result 
