
function model = LDA_train(TrainingData, TrainingLabel)
%
% TrainingData: 
%               m*n matrix
%               m is quantity of data
%               n is quantity of features
% TrainingLabel:
%               vector
%               1 or 0
%
% Copyright(c) 2012
% State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University
% Written by <a href="zaixucui@gmail.com">Zaixu Cui</a>
% Mail to Author:  <a href="zaixucui@gmail.com">zaixucui@gmail.com</a>
%

[rows_quantity, feature_quantity] = size(TrainingData);
group1_quantity = length(find(TrainingLabel == 1));
group2_quantity = length(find(TrainingLabel == -1));
Training_group1 = TrainingData(find(TrainingLabel == 1), :);
Training_group1 = Training_group1';
Training_group2 = TrainingData(find(TrainingLabel == -1), :);
Training_group2 = Training_group2';

% LDA classification
Sw = zeros(feature_quantity);
group1_average = mean(Training_group1, 2);
group2_average = mean(Training_group2, 2);
for j = 1:group1_quantity
    Sw = Sw + (Training_group1(:, j) - group1_average) * (Training_group1(:, j) - group1_average)';
end
for j = 1:group2_quantity
    Sw = Sw + (Training_group2(:, j) - group2_average) * (Training_group2(:, j) - group2_average)';
end

w = inv(Sw) * (group1_average - group2_average);

Sum = 0;
for j = 1:group1_quantity
    Y_group1(j) = w' * (Training_group1(:, j));
end
for j = 1:group2_quantity
    Y_group2(j) = w' * (Training_group2(:, j));
end
Y_group1_Average = mean(Y_group1);
Y_group2_Average = mean(Y_group2);
% b = -(Y_group1_Average + Y_group2_Average) / 2 + log(Training_group1_quantity / Training_group2_quantity);
b = (group1_quantity * Y_group1_Average + group2_quantity * Y_group2_Average) / (group1_quantity + group2_quantity);

model.w = w;
model.group1_average = Y_group1_Average;
model.group1_label = 1;
model.group2_average = Y_group2_Average;
model.group2_label = -1;
model.b = b;