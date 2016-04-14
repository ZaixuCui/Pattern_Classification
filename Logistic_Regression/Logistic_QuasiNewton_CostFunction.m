
function Cost = Logistic_QuasiNewton_CostFunction(theta)

load TrainingData.mat;
load TrainingLabel.mat;

[SubjectQuantity FeatureQuantity] = size(TrainingData);

group1_Index = find(TrainingLabel);
group0_Index = find(~TrainingLabel);

hypothesis = 1 ./ (1 + exp(-TrainingData  * theta));
Cost = sum(log(hypothesis(group1_Index))) + sum(log(1 - hypothesis(group0_Index)));
Cost = -Cost / SubjectQuantity;

