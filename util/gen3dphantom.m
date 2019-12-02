%Script to generate the 3D phantom.
%-Take a 3D fft and sum of squares coil combine the raw 3D knee data to get a ground truth image
%-Compute ESPIRIT coil sensitivity maps so that we can generate wave-encoded data from the ground truth image and coil sensitivity maps.

fprintf('Loading data... ')
kspace = readcfl('../phantom/knee3d');
fprintf('done\n')

[M,N,P,C] = size(kspace);

fprintf('Dimensions of kspace: [%d,%d,%d,%d]\n',M,N,P,C)

fprintf('Computing ground truth image... ')
gt     = bart('rss 8',bart('fft -i 7',kspace));
fprintf('done\n')

fprintf('Computing coil sensitivity maps... ')
maps   = bart('ecalib -m 1',kspace);
