%input_layer_size = 2;
%hidden_layer_size = 2;
%num_labels = 4;
%nn_params = [ 1:18 ] / 10;
%X = cos([1  2 ; 3  4 ; 5  6]);
%y = [4; 2; 3];
%lambda = 0;
%feedforward = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% output:
% ans =  7.4070

%input_layer_size = 2;
%hidden_layer_size = 2;
%num_labels = 4;
%nn_params = [ 1:18 ] / 10;
%X = cos([1  2 ; 3  4 ; 5  6]);
%y = [4; 2; 3];
%lambda = 3;
%regularised = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%output:
%ans =  16.457

% Test 2b:
% input:
%input_layer_size = 2;
%hidden_layer_size = 2;
%num_labels = 4;
%nn_params = [ 1:18 ] / 10;
%X = cos([1  2 ; 3  4 ; 5  6]);
%y = [4; 2; 3];
%lambda = 4;
%nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% output:
% ans =  19.474


% ---------------------------------------
% Test 3a (Sigmoid Gradient):
% input:
%sigmoidGradient(1);

% output:
% ans =  0.19661
% 
% -------
% Test 3b:
% input:
%sigmoidGradient([2 3]);

% output:
% ans =
% 
%    0.104994   0.045177
% 
% -------
% Test 3c:
% input:
%sigmoidGradient([-2 0; 4 999999; -1 1]);

% output:
% ans =
% 
%    0.10499   0.25000
%    0.01766   0.00000
%    0.19661   0.19661


% ---------------------------------------
% Test 4a (Neural Network Gradient (Backpropagation)):
% input:
%input_layer_size = 2;
%hidden_layer_size = 2;
%num_labels = 4;
%nn_params = [ 1:18 ] / 10;
%X = cos([1  2 ; 3  4 ; 5  6]);
%y = [4; 2; 3];
%lambda = 0;
%[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% output:
% J =  7.4070
% grad =
% 
%    0.766138
%    0.979897
%   -0.027540
%   -0.035844
%   -0.024929
%   -0.053862
%    0.883417
%    0.568762
%    0.584668
%    0.598139
%    0.459314
%    0.344618
%    0.256313
%    0.311885
%    0.478337
%    0.368920
%    0.259771
%    0.322331
% 
% ---------------------------------------
% Test 5a (Regularized Gradient):
% input:
input_layer_size = 2;
hidden_layer_size = 2;
num_labels = 4;
nn_params = [ 1:18 ] / 10;
X = cos([1  2 ; 3  4 ; 5  6]);
y = [4; 2; 3];
lambda = 3;
[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% output:
% J =  16.457
% grad =
% 
%    0.76614
%    0.97990
%    0.27246
%    0.36416
%    0.47507
%    0.54614
%    0.88342
%    0.56876
%    0.58467
%    0.59814
%    1.55931
%    1.54462
%    1.55631
%    1.71189
%    1.97834
%    1.96892
%    1.95977
%    2.12233
% 
% -------
% Test 5b:
% input:
input_layer_size = 2;
hidden_layer_size = 2;
num_labels = 4;
nn_params = [ 1:18 ] / 10;
X = cos([1  2 ; 3  4 ; 5  6]);
y = [4; 2; 3];
lambda = 4;
[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% output:
% J =  19.474
% grad =
% 
%    0.76614
%    0.97990
%    0.37246
%    0.49749
%    0.64174
%    0.74614
%    0.88342
%    0.56876
%    0.58467
%    0.59814
%    1.92598
%    1.94462
%    1.98965
%    2.17855
%    2.47834
%    2.50225
%    2.52644
%    2.72233
