function out = addnoise(in,stdn,mask)
    dims = size(in);
    out = in + mask.*(stdn*(randn(dims) + 1i*randn(dims)));
end
