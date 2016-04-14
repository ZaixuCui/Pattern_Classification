
function Permutation_test_1(OriginBrain, RandBrainCell, P_Threshold)

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

RandValues = [];
for i = 1:RandQuantity
    disp(i);
    tmp = load(RandBrainCell{i}); 
    RandValues = [RandValues; tmp.w_Brain];
end

for j = 1:length(OriginValue.w_Brain)
    tmp = RandValues(:, j);
    P_Brain(j) = length(find(abs(tmp) >= abs(OriginValue.w_Brain(j)))) / RandQuantity;
end

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

