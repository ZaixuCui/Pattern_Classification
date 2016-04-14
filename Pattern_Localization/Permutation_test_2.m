
function Permutation_test_2(OriginBrain, RandBrainCell, P_Threshold)

%
% OriginBrain:
%         path of .mat file, 
%         a variable named 'w_Brain' should be stored in the mat file
%
% RandBrainCell:
%         cell of paths of .mat files
%         a variable named 'w_Brain' should be stored in each mat file
%
% P_Threshold:
%         vector of p threshold
%         such as [ 0.05 0.01 ] 
%

OriginValue = load(OriginBrain);
RandQuantity = length(RandBrainCell);

% RandValues = [];
w_Brain_tmp = cell(RandQuantity, 1);
for i = 1:RandQuantity
    disp(i);
    tmp = load(RandBrainCell{i}); 
    [rows_Quantity, ~] = size(tmp.w_Brain);
    if rows_Quantity == 1;
        w_Brain_tmp{i} = tmp.w_Brain;
    else
        w_Brain_tmp{i} = tmp.w_Brain';
    end
end
% RandValues = [RandValues; tmp.w_Brain];

P_Brain = zeros(1, length(tmp.w_Brain));
    
QuantityForEachFolder = 100;
SplitQuantity = RandQuantity / QuantityForEachFolder;
for i = 1:SplitQuantity   
    disp(i);
    clear RandValues;
    RandValues = cell2mat(w_Brain_tmp([QuantityForEachFolder * (i-1) + 1 : QuantityForEachFolder * i]));
    for j = 1:length(OriginValue.w_Brain)
        tmp = RandValues(:, j);
        P_Brain_tmp(j) = length(find(abs(tmp) >= abs(OriginValue.w_Brain(j))));
    end
    P_Brain = P_Brain + P_Brain_tmp;
end
P_Brain = P_Brain / RandQuantity;

[OriginFolder, y, z] = fileparts(OriginBrain);
save([OriginFolder filesep 'p_Brain.mat'], 'P_Brain');

for i = 1:length(P_Threshold)
    
    clear w_Brain;
    w_Brain = OriginValue.w_Brain;
    w_Brain(find(P_Brain >= P_Threshold(i))) = 0;
    P_Threshold_String = num2str(P_Threshold(i));
    P_Threshold_String = strrep(P_Threshold_String, '.', '');
    save([OriginFolder filesep 'w_Brain_' P_Threshold_String '.mat'], 'w_Brain');
    
end

