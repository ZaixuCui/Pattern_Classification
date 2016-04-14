
function PatternLocalization_MainFunction(Origin_Accuracy_Path, Rand_Accuracy_Path_Cell, Origin_w_Brain, Rand_w_Brain_Cell)

[ResultantFolder y z] = fileparts(Origin_Accuracy_Path);

disp('display accuracy distribution in random state:');
Accuracy_Distribution(Rand_Accuracy_Path_Cell, ResultantFolder);

disp('Claculate accuracy siginificance:');
Accuracy_Significance_Acc(Origin_Accuracy_Path, Rand_Accuracy_Path_Cell);

disp('display w distribution in random state:');
W_Distribution(Rand_w_Brain_Cell, 50, ResultantFolder);

disp('Permutation test for w Brain:');
Permutation_test(Origin_w_Brain, Rand_w_Brain_Cell, 0.05);