function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


n = max(size(theta)); % number of features

% Calculate P(y=1|x,theta) for training set.
% theta [n x 1]
% X [m x n]
% y [m x 1]
% h(theta) = sigmoid(X * theta) is vector [m x 1].
prob = sigmoid(X * theta);

for i=1:m
    J = J + (-y(i) * log(prob(i)) - (1 - y(i)) * log(1 - prob(i)));
end

J = J/m;

% Adds reularization to const function.
% Remember theta(1) should not be regularized!
J = J + lambda * sum(theta(2:end).^2) / (2 * m);

% Calculate gradients.
% Remember theta(1) should not be regularized!
for j=1:n
    for i=1:m
        grad(j) = grad(j) + (prob(i) - y(i)) * X(i,j);
    end

    % Adds reularization to const function.
    if (j > 1)
        grad(j) = grad(j) + lambda * theta(j);
    end

    grad(j) = grad(j) / m;
end


% =============================================================

end
