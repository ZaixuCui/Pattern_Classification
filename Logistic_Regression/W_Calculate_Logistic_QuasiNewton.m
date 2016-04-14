function W_Calculate_Logistic_QuasiNewton(Subjects_Data, Subjects_Label, Pre_Method, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 0 or 1
%
% Pre_Method:
%           'Normalize' or 'Scale'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[~, Feature_Quantity] = size(Subjects_Data);

t=weka.classifiers.functions.Logistic();
% tmp=wekaArgumentString({'-R', 0});
% t.setOptions(tmp);

% if strcmp(Pre_Method, 'Normalize')
%     %Normalizing
%     MeanValue = mean(Subjects_Data);
%     StandardDeviation = std(Subjects_Data);
%     [~, columns_quantity] = size(Subjects_Data);
%     for j = 1:columns_quantity
%         Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
%     end
% elseif strcmp(Pre_Method, 'Scale')
%     % Scaling to [0 1]
%     MinValue = min(Subjects_Data);
%     MaxValue = max(Subjects_Data);
%     [~, columns_quantity] = size(Subjects_Data);
%     for j = 1:columns_quantity
%         Subjects_Data(:, j) = (Subjects_Data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
%     end
% end

% Logistic classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
Subjects_Data = double(Subjects_Data);

X_Y = data(Subjects_Data, Subjects_Label);
cat = wekaCategoricalData(X_Y);
t.buildClassifier(cat);

x=1;
    
