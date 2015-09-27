function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.005;
temp_C = C;
error = 1000000;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
for i=1:10
  temp_sigma = 0.005;
  temp_C = temp_C * 3;
  for j=1:10
    temp_sigma = temp_sigma * 3;
    predictions =svmPredict(@svmTrain(X, y, temp_C, @(x1,x2) gaussianKernel(x1,x2,temp_sigma)), Xval);
    new_error = mean(double(predictions ~= yval));
    if(new_error < error)
      C = temp_C;
      sigma = temp_sigma;
      error = new_error;
    end
  end
end
C
sigma






% =========================================================================

end
