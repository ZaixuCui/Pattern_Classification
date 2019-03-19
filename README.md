# Pattern_Classification
Matlab code of pattern classification algorithms (e.g., SVC, logistic regression, MLDA) for analyzing brain imaging data.

Citing our related paper will be greatly appreciated if you use these codes.
<br>&emsp; ```Cui, Z. , Xia, Z. , Su, M. , Shu, H. and Gong, G. (2016), Disrupted white matter connectivity underlying developmental dyslexia: A machine learning approach. Hum. Brain Mapp., 37: 1443-1458. doi:10.1002/hbm.23112```

Copyright (c) Zaixu Cui, State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University.  
Contact information: zaixucui@gmail.com



SVM was implemented using LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/).
Dowload it and compile the matlab version in MATLAB. Just run make.m file to compile.
If installation is correct, then input 'svmtrain' in the MATLAB command window, will have the following output:

```Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');
libsvm_options:
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	4 -- precomputed kernel (kernel values in training_instance_matrix)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)```
