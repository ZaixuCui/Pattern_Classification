
function Accuracy_Significance_Acc(OriginalAccuracyPath, RandAccuracyPathCell)
%
% OriginalAccuracyPath:
%           path of .mat file
%
% RandAccuracyPathCell:
%           cell of paths of .mat files         
%

tmp = load(OriginalAccuracyPath);
OriginalAccuracy = tmp.Accuracy;

BiggerQuantity = 0; % the quantity of random cases, which correctly 
                    % classify more samples than original case, or equal to
                    % original case 
for i = 1:length(RandAccuracyPathCell)
    tmp = load(RandAccuracyPathCell{i});
    RandAccuracy = tmp.Accuracy;
    if RandAccuracy >= OriginalAccuracy
        BiggerQuantity = BiggerQuantity + 1;
    end
end

P_value = BiggerQuantity / length(RandAccuracyPathCell);
[OriginalFolder, y , z] = fileparts(OriginalAccuracyPath);
save([OriginalFolder filesep 'Accuracy_Significance.mat'], 'P_value');
