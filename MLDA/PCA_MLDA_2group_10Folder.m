function [Accuracy Sensitivity Specificity Category] = PCA_MLDA_2group_10Folder(Subjects_Data, Subjects_Label, type, Parameters, ResultantFolder)

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

if nargin >= 5
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

Subjects_Quantity = length(Subjects_Label);
[Splited_Data, Splited_Data_Label, Origin_ID_Cell] = Split_NFolders(Subjects_Data, Subjects_Label, 10);

predicted_labels = [];
decision_values = [];

for i = 1:10
    
    disp(['The ' num2str(i) ' iteration!']);
    
    % Select training data and testing data
    test_label = Splited_Data_Label{i};
    test_data = Splited_Data{i};
    
    Training_all_data = [];
    Label = [];
    for j = 1:10
        if j == i
            continue;
        end
        Training_all_data = [Training_all_data; Splited_Data{j}];
        Label = [Label; Splited_Data_Label{j}];
    end

    % Normalizing
    MeanValue = mean(Training_all_data);
    StandardDeviation = sqrt(var(Training_all_data));
    [rows, columns_quantity] = size(Training_all_data);
    for j = 1:columns_quantity
        if StandardDeviation(j)
            Training_all_data(:, j) = (Training_all_data(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    end
    [COEFF, SCORE, latent, tsquare] = princomp(Training_all_data, 'econ');

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
    for j = 1:length(Label)
        PCA_All(j, :) = SCORE(j, 1:ComponentQuantity);
    end

    % SVM classification
    Label = reshape(Label, length(Label), 1);
    PCA_All = double(PCA_All);
    model(i) = MLDA_train(PCA_All, Label);
    
    % NC_female is 1, Nonmosaic is 0
    MeanValue_New = repmat(MeanValue, length(test_label), 1);
    StandardDeviation_New = repmat(StandardDeviation, length(test_label), 1);
    test_data = (test_data - MeanValue_New) ./ StandardDeviation_New;
    test_data = test_data * COEFF;
    test_data = test_data(:, [1:ComponentQuantity]);

    % predicts
    test_data = double(test_data);   
    for j = 1:length(test_label)
        predicted_labels_tmp(j) = MLDA_predict(model(i), test_data(j, :), test_label(j));
    end
    predicted_labels = [predicted_labels predicted_labels_tmp];
    clear predicted_labels_tmp;
    
end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
% Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
% Y_group0 = decision_values(Group0_Index);

if nargin >= 5
    % save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
    save([ResultantFolder filesep 'Category.mat'], 'Category_group1', 'Category_group0');
end

Category.Category_group0 = Category_group0;
Category.Category_group1 = Category_group1;

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == -1);
group1_Wrong_Quantity = length(group1_Wrong_ID);
Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;
Sensitivity = (length(Group0_Index) - group0_Wrong_Quantity) / length(Group0_Index);
Specificity = (length(Group1_Index) - group1_Wrong_Quantity) / length(Group1_Index);

if nargin >= 5
    disp(['group0: ' num2str(group0_Wrong_Quantity) ' subjects are wrong ' mat2str(group0_Wrong_ID) ]);
    disp(['group1: ' num2str(group1_Wrong_Quantity) ' subjects are wrong ' mat2str(group1_Wrong_ID) ]);
    save([ResultantFolder filesep 'WrongInfo.mat'], 'group0_Wrong_Quantity', 'group0_Wrong_ID', 'group1_Wrong_Quantity', 'group1_Wrong_ID');
    disp(['Accuracy is ' num2str(Accuracy) ' !']);
    save([ResultantFolder filesep 'Accuracy.mat'], 'Accuracy');
    disp(['Sensitivity is ' num2str(Sensitivity) ' !']);
    save([ResultantFolder filesep 'Sensitivity.mat'], 'Sensitivity');
    disp(['Specificity is ' num2str(Specificity) ' !']);
    save([ResultantFolder filesep 'Specificity.mat'], 'Specificity');

    % Averaging w
    w_average = 0;
    [tmp w_length] = size(COEFF);
    for i = 1:Subjects_Quantity
        tmp = zeros(1, w_length);
        tmp(1:length(model(i).w)) = tmp(1:length(model(i).w)) + model(i).w';
        w_average = w_average + tmp;
    end
    w_average = w_average / Subjects_Quantity;
    w_Brain = w_average * COEFF';
    save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');
end