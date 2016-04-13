
function Permutation_test_pMap(OriginBrain, P_Brain, P_Threshold)
%
% OriginBrain:
%         path of .mat file, 
%         a variable named 'w_Brain' should be stored in the mat file
%
% P_Brain:
%         image of P map, should be converted a vector
%         can be acquired by load_nii command
%         e.g., tmp = load_nii(PMapPath);
%               Mask_nii = load_nii(MaskPath);
%               P_Brain = tmp.img(Mask_nii.img == 1);
%                             
% P_Threshold:
%         such as 0.05, 0.01, ...
%

OriginValue = load(OriginBrain);

for i = 1:length(P_Brain)
    
    if P_Brain(i) >= P_Threshold
        OriginValue.w_Brain(i) = 0;
    end
    
end

[OriginFolder, y, z] = fileparts(OriginBrain);
w_Brain = OriginValue.w_Brain;
P_Threshold_String = num2str(P_Threshold);
P_Threshold_String = strrep(P_Threshold_String, '.', '');
save([OriginFolder filesep 'w_Brain_' P_Threshold_String '.mat'], 'w_Brain');