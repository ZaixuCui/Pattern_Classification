
function theta = Logistic_QuasiNewton_train(TrainingData, TrainingLabel, ridge)

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

hypothesis = 1 ./ (1 + exp(-TrainingData  * theta));
gradient_old = ((1 - hypothesis) .* TrainingLabel)' * TrainingData - ridge * theta';

H = eye(thetaSize);

for k = 1:10
% delta_theta = ones(thetaSize, 1);
% delta_gradient = zeros(thetaSize, 1);
% while sum(delta_theta - H * delta_gradient) >= 0.000001
%     u = -H * gradient_old';
    delta_theta = (gradient_old * u) / (u' * H * u) * u;
    theta = theta - delta_theta;
    
    gradient_new = ((1 - hypothesis) .* TrainingLabel)' * TrainingData - ridge * theta';
    delta_gradient = (gradient_new - gradient_old)';
    gradient_old = gradient_new;
    
    b = 1 + (delta_gradient' * H * delta_gradient) / (delta_theta' * delta_gradient);
    H = H + (1 / (delta_theta' * delta_gradient)) * (b * delta_theta * delta_theta' - ...
        delta_theta * delta_gradient' * H - H * delta_gradient * delta_theta');
end
% end


x=1;

% % options = optimoptions(@fminunc, 'Allgorithm', 'quasi-newton');
% save TrainingData.mat TrainingData;
% save TrainingLabel.mat TrainingLabel;
% options = optimset('Display', 'iter', 'Algorithm', 'sqp');
% [theta, fval, exitflag, output] = fminunc(@Logistic_QuasiNewton_CostFunction, theta, options);


