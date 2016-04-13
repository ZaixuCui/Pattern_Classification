
function Accuracy_Significance(OriginalCategoryPath, RandCategoryPathCell)
%
% OriginalCategoryPath:
%           path of .mat file
%
% RandCategoryPathCell:
%           cell of paths of .mat files         
%
% Two variables named 'Category_group0' and 'Category_group1' should be 
% stored in the mat file.
% '-1' in 'Category_group0' indicates correctly classified
% '1' in 'Category_group0' indicates wrongly classified
%

OriginalCategory = load(OriginalCategoryPath);
OriginalCorrectQuantity = length(find(OriginalCategory.Category_group0 == -1)) + ...
    length(find(OriginalCategory.Category_group1 == 1));
disp(OriginalCorrectQuantity);

BiggerQuantity = 0; % the quantity of random cases, which correctly 
                    % classify more samples than original case, or equal to
                    % original case 
for i = 1:length(RandCategoryPathCell)
    RandCategory = load(RandCategoryPathCell{i});
    RandCorrectQuantity = length(find(RandCategory.Category_group0 == -1)) + ...
        length(find(RandCategory.Category_group1 == 1));
    disp(RandCorrectQuantity);
    if RandCorrectQuantity >= OriginalCorrectQuantity
        BiggerQuantity = BiggerQuantity + 1;
    end
end

P_value = BiggerQuantity / length(RandCategoryPathCell);
[OriginalFolder, y , z] = fileparts(OriginalCategoryPath);
save([OriginalFolder filesep 'Accuracy_Significance.mat'], 'P_value');
