function Vector_to_Matrix(Vector_Brain, ROIQuantity, ResultantFile)

%
% Vector_Brain:
%        w vector calculated during classfication
%
% ROIQuantity:
%        quantity of ROIs in the altas
%
% ResultantFile:
%        the path of resultant file (.mat)
%


tmp = magic(ROIQuantity);
TriuMatrix = triu(tmp, 1);
TriuIndex = find(TriuMatrix ~= 0);
w_Brain_Matrix = zeros(ROIQuantity, ROIQuantity);
w_Brain_Matrix(TriuIndex) = Vector_Brain;
save(ResultantFile, 'w_Brain_Matrix');