
function [label, predict_value] = MLDA_predict(model, test_data, test_label)
%
% model: output of MLDA_train
% test_data: n*1 vector
%
% Copyright(c) 2012
% State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University
% Written by <a href="zaixucui@gmail.com">Zaixu Cui</a>
% Mail to Author:  <a href="zaixucui@gmail.com">zaixucui@gmail.com</a>
%
predict_value = model.w' * test_data';
if model.group1_average <= model.group2_average
    if predict_value <= model.b
        label = model.group1_label;
    else
        label = model.group2_label;
    end
else
    if predict_value <= model.b
        label = model.group2_label;
    else
        label = model.group1_label;
    end
end

if label == test_label
    disp('100%');
else
    disp('0%');
end

predict_value = predict_value - model.b;