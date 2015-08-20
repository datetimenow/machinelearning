function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1) X];
z2 = sigmoid(a1 * transpose(Theta1));
a2 = [ones(size(z2,1),1) z2];
a3 = sigmoid(a2 * transpose(Theta2));
E = eye(num_labels);
y1 = E(y,:);
J = sum(sum(- log(a3) .* y1- log(1 - a3) .* (0 == y1)))/m;
d3 = a3 - y1;
d2 = (d3 * Theta2(:,2:end)).*z2.*(1-z2);
a = size(a1);
temp1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
temp2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = transpose(d2)*a1/m + temp1 * lambda / m;
Theta2_grad = transpose(d3)*a2/m + temp2 * lambda / m;

thetas1 = size(Theta1);
thetas2 = size(Theta2);
Xs = size(X);
thetagrad1 = size(Theta1_grad);
thetagrad2 = size(Theta2_grad);
temp1 = ones(input_layer_size+1,1);
temp1(1) = 0;
temp2 = ones(hidden_layer_size,1);
%temp2(1) = 0;
temp3 = ones(hidden_layer_size+1,1);
temp3(1) = 0;
temp4 = ones(num_labels,1);
%temp4(1) = 0;    

t1 = Theta1.^2 * temp1;
t11 = transpose(t1) * temp2;
t2 = Theta2.^2 * temp3;
t21 = transpose(t2) * temp4;

J = J + ((t11 + t21) * lambda / (2 * m));
%for t = 1:m
%	a_1 = [1 X(t,:)];
%	z_2 = sigmoid(a_1 * transpose(Theta1));
%	a_2 = [1 z_2];
%	a_3 = sigmoid(a_2 * transpose(Theta2));
%	d_3 = (a_3 - y1(t,:));
%	%a = size(a_1)
%	%z = size(z_2)
%	%d = size(d_3)
%	%a2 = size(a_2)
%	a3 = size(a_3);
%	d_2 = d_3*Theta2(:,2:end).*(a_2(:,2:end).*(1-a_2(:,2:end)));
%	%n1 = size(transpose(d_2)*a_1)
%	%n2 = size(transpose(d_3)*a_2)
%	Theta1_grad = Theta1_grad + transpose(d_2)*a_1;
%	Theta2_grad = Theta2_grad + transpose(d_3)*a_2;
%end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
