
function [label, predict_value] = Logistic_predict(theta, test_data, test_label)

%
% theta:
%       m + 1 * 1 vector
%
% test_data:
%       n * m matrix, n is quantity of test subject, m is quantity of
%       features
%
% test_label:
%       n * 1 vector, 0 or 1
%
[~, FeaturesQuantity] = size(test_data);
[~, theta_col] = size(theta);
if theta_col ~= 1
    error('The quantity of columns of the first parameter should be 1.');
end

[test_data_quantity, ~] = size(test_data);
test_data = [ones(test_data_quantity, 1) test_data];
for i = 1:test_data_quantity
    predict_value(i) = 1 / (1 + exp(-test_data(i, [2:FeaturesQuantity + 1])  * theta))
    if predict_value(i) >= 0.5
        label(i) = 1;
    else
        label(i) = 0;
    end

    if label(i) == test_label(i)
        disp('100%');
    else
        disp('0%');
    end
end
