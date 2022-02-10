function PCA_SVM_2group(Subjects_Data, Subjects_Label, type, ResultantFolder, Parameters)

%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 1 or -1
%
% type:
%           'ComponentQuantity'
%           'CumulativeVariation'
%           'BiggerThanMean'
% 
% Parameters:
%           the quantity of components when type is 'ComponentQuantity';
%           the proportion of cumulative variation when type is
%           'CumulativeVariation'
%           don't need when type is 'BiggerThanMean'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[Subjects_Quantity Features_Quantity] = size(Subjects_Data);

for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Subjects_Data_tmp = Subjects_Data;
    Subjects_Label_tmp = Subjects_Label;
    % Select training data and testing data
    test_label = Subjects_Label_tmp(i);
    test_data = Subjects_Data_tmp(i, :);
    
    Subjects_Label_tmp(i) = [];
    Subjects_Data_tmp(i, :) = [];
    Training_group1_Index = find(Subjects_Label_tmp == 1);
    Training_group0_Index = find(Subjects_Label_tmp == -1);
    Training_group1_data = Subjects_Data_tmp(Training_group1_Index, :);
    Training_group0_data = Subjects_Data_tmp(Training_group0_Index, :);
    Training_group1_Label = Subjects_Label_tmp(Training_group1_Index);
    Training_group0_Label = Subjects_Label_tmp(Training_group0_Index);
    
    % PCA for training data
    All_Training = [Training_group1_data; Training_group0_data];
    Label = [Training_group1_Label Training_group0_Label];

    % Normalizing
    MeanValue = mean(All_Training);
    StandardDeviation = std(All_Training);
    [rows, columns_quantity] = size(All_Training);
    for j = 1:columns_quantity
        if StandardDeviation(j)
            All_Training(:, j) = (All_Training(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    end
    [COEFF, SCORE, latent, tsquare] = princomp(All_Training, 'econ');

    if strcmp(type, 'ComponentQuantity')
        ComponentQuantity = Parameters;
    elseif strcmp(type, 'CumulativeVariation')
        ComponentQuantity = length(latent);
        while 1
            if sum(latent(1:ComponentQuantity)) / sum(latent) <= Parameters
                break;
            else
                ComponentQuantity = ComponentQuantity - 1;
            end
        end
    elseif strcmp(type, 'BiggerThanMean')
        tmp = find(latent >= mean(latent));
        ComponentQuantity = tmp(end);
    end
    
    clear PCA_All;
    for j = 1:Subjects_Quantity - 1
        PCA_All(j, :) = SCORE(j, 1:ComponentQuantity);
    end

    % SVM classification
    Label = reshape(Label, length(Label), 1);
    PCA_All = double(PCA_All);
    model(i) = svmtrain(Label, PCA_All,'-t 0');
    
    % NC_female is 1, Nonmosaic is 0
    test_data = (test_data - MeanValue) ./ StandardDeviation;
    test_data = test_data * COEFF;
    test_data = test_data(1:ComponentQuantity);

    % predicts
    test_data = double(test_data);
    [predicted_labels(i), accuracy, tmp] = svmpredict(test_label, test_data, model(i));
    
    % Calculate decision value
    w{i} = zeros(1, ComponentQuantity);
    for j = 1 : model(i).totalSV
        w{i} = w{i} + model(i).sv_coef(j) * model(i).SVs(j, :);
    end
    decision_values(i) = w{i} * test_data' - model(i).rho;
%     decision_values(i) = w{i} * test_data';
    
%     % Back w to brain
%     w_Brain_PCA = zeros(1, length(latent));
%     w_Brain_PCA(1:length(w{i})) = w_Brain_PCA(1:length(w{i})) + w{i};
%     w_Brain_Cell{i} = w_Brain_PCA * COEFF';

end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
Y_group0 = decision_values(Group0_Index);

save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
save([ResultantFolder filesep 'Category.mat'], 'Category_group1', 'Category_group0');

% % Averaging w
% [tmp w_length] = size(COEFF);
% w_average = zeros(1, w_length);
% for i = 1:Subjects_Quantity
%     tmp = zeros(1, w_length);
%     tmp(1:length(w{i})) = tmp(1:length(w{i})) + w{i};
%     w_average = w_average + tmp;
% end
% w_average = w_average / Subjects_Quantity;
% w_Brain = w_average * COEFF';

% w_Brain_Mat = cell2mat(w_Brain_Cell');
% for i = 1:Features_Quantity
%     [minum index] = min(abs(w_Brain_Mat(:, 1)));
%     w_Brain(i) = w_Brain_Mat(index, i);
% end

% Calculate w using all subjects
W_Calculate_PCA_SVM(Subjects_Data, Subjects_Label, type, ResultantFolder, Parameters);

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == -1);
group1_Wrong_Quantity = length(group1_Wrong_ID);
disp(['group0: ' num2str(group0_Wrong_Quantity) ' subjects are wrong ' mat2str(group0_Wrong_ID) ]);
disp(['group1: ' num2str(group1_Wrong_Quantity) ' subjects are wrong ' mat2str(group1_Wrong_ID) ]);
save([ResultantFolder filesep 'WrongInfo.mat'], 'group0_Wrong_Quantity', 'group0_Wrong_ID', 'group1_Wrong_Quantity', 'group1_Wrong_ID');
Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;
disp(['Accuracy is ' num2str(Accuracy) ' !']);
save([ResultantFolder filesep 'Accuracy.mat'], 'Accuracy');
