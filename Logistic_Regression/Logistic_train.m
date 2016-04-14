
function theta = Logistic_train(TrainingData, TrainingLabel, alpha)

%
% TrainingData:
%        n * m vector
%        n is quantity of subjects
%        m is quantity of features
%
% TrainingLabel:
%        vector of 0 or 1, n * 1 vector
%
% alpha:
%        learning rate, should be neither too small nor too big 
%        0~1
%
% Output
% theta:
%        (n + 1) * 1 vector
%

[SubjectQuantity FeatureQuantity] = size(TrainingData);

% Add one column which is all ones to the first cloumn of X, which is for theta(0).
X_Zero = ones(SubjectQuantity, 1);
TrainingData = [X_Zero TrainingData];

% Check parameters
[Label_RowQuantity Label_ColumnQuantity] = size(TrainingLabel);
if Label_RowQuantity ~= SubjectQuantity 
    error('The first parameter should have the same quantity of rows as the second parameter.');
end
if Label_ColumnQuantity ~= 1
    error('The second parameter should contain only one column.');
end

% Initialize theta
thetaSize = FeatureQuantity + 1;
theta = zeros(thetaSize, 1);

% Iteration for fitting parameter theta
TerminateSlop = 0.0001;
Slop = Inf;
IterationNum = 0;
MaxIteration = 1000;
Cost = 1;

group1_Index = find(TrainingLabel);
group0_Index = find(~TrainingLabel);

while Cost > TerminateSlop & IterationNum < MaxIteration
    
    hypothesis = 1 ./ (1 + exp(-TrainingData  * theta));
    update = alpha * (hypothesis - TrainingLabel)' * TrainingData;
    
    theta = theta - update';
  
%     Slop = sum(abs(update));
    
    IterationNum = IterationNum + 1;
    
    % If cost is close to 0, then stop
    Cost = sum(log(hypothesis(group1_Index))) + sum(log(1 - hypothesis(group0_Index)));
    Cost = -Cost / SubjectQuantity;
    Cost_Record(IterationNum) = Cost;
    
end
plot(Cost_Record);
display(sprintf('iter times;%d\tCost：%6.5f\n', IterationNum, Cost));
