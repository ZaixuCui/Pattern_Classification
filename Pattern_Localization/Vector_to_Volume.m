function Vector_to_Volume(Vector_Brain, Mask_path, ResultantFile)

%
% w_Brain:
%        w vector calculated during classfication
%
% Mask_path:
%        path of brain mask image
%
% ResultantFile:
%        the path of resultant file
%

Mask_hdr = spm_vol(Mask_path);
Mask_data = spm_read_vols(Mask_hdr);
[ResultantFolder, ~, ~] = fileparts(ResultantFile);
if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end
% Back to brain
Index = find(Mask_data == 1);
Discriminating_Volume = zeros(size(Mask_data));
[x y z] = ind2sub(size(Mask_data), Index);
for i = 1:length(x)
    Discriminating_Volume(x(i), y(i), z(i)) = Vector_Brain(i);
end
hdr = spm_vol(Mask_path);
hdr.fname = ResultantFile;
hdr.dt = [16 0];
spm_write_vol(hdr, Discriminating_Volume);
