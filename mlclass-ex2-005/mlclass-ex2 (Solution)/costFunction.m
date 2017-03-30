function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Calculate P(y=1|x,theta) for all training set.
% theta [n x 1]
% X [m x n]
% y [m x 1]
% h(theta) = sigmoid(X * theta) is vector [m x 1].
prob = sigmoid(X * theta);

for i=1:m
    J = J + (-y(i) * log(prob(i)) - (1 - y(i)) * log(1 - prob(i)));
end;

J = J/m;

n = max(size(theta));

for j=1:n
    for i=1:m
        grad(j) = grad(j) + (prob(i) - y(i)) * X(i,j);
    end;
    grad(j) = grad(j) / m;
end;

% =============================================================

end
