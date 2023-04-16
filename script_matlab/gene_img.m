clear;
close;

path = '../output/sino/';
files = dir(fullfile(path, '*.nii'));
filenames = {files.name}';
for i = 1:length(filenames)
    sin_img = double(niftiread(strcat(path, filenames{i})));
    recon = iradon(sin_img, gene_angle(720), 256);
    niftiwrite(recon, strcat('../output/img/scope_', filenames{i}))
end
