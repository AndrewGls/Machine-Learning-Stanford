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

% Theta1 is matrix [25 x 401] for layer 2.
% Theta2 is matrix [10 x 26] for layer 3.

% Convert y to vector[10,1] for cont function calculation.
Y = zeros(m, num_labels);
for i=1:m
    num = y(i);
    Y(i,num) = 1;
end


% >>>>>>>>>>>>>>>>>>>>>>>
% Part 1 - Feedforward

% Adds extra bias unit to X
X = [ones(m, 1) X];

% Calculate activation function for Layer 2
z2 = X * Theta1';
a2 = sigmoid(z2);
             
% Layer 3
% Adds extra bias neuron to layer a2
a2 = [ones(size(a2, 1), 1) a2];
a3 = sigmoid(a2 * Theta2');

% Calculate cost function.
J = sum(sum((-Y.*log(a3) - (1 - Y).*log(1-a3)), 2)) / m;

% Adds regularization into the cont function.
% Note that you should not be regularizing the terms that correspond to the bias.
% Remove bias from Theta1 and Theta2.
reg = lambda * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) / (2 * m);
J = J + reg;

             
% >>>>>>>>>>>>>>>>>>>>>>>
% Part 2 - Backpropagation algorithm

for i=1:m

% Feedforward pass for training example i.
             
    a1 = X(i,:)';      % vector (401,1)
    z2 = Theta1 * a1;  % vector (25,1)
    a2 = sigmoid(z2);  % vector (25,1)
             
    %calculate a3
    a2 = [1; a2];      % vector (26,1)
    z3 = Theta2 * a2;  % vector (10,1)
    a3 = sigmoid(z3);  % vector (10,1)
         
    % calculate delta3
    % extract vector y(i) for example i.
    Yi = Y(i,:)';      % vector (10,1)
    delta3 = a3 - Yi;  % vector (10,1)

    % calculate delta2 as vector (25,1)
    delta2 = Theta2' * delta3 .* sigmoidGradient([1; z2]);  % vector (26,1)
    %Remove bias from delta2 - column 1.
    delta2 = delta2(2:end);     % vector (25,1)

    % accumulate the gradient for layer 2 as matrix (10x26)
    Theta2_grad = Theta2_grad + delta3 * a2';

    % accumulate the gradient for layer 1 as matrix (25x401)
    Theta1_grad = Theta1_grad + delta2 * a1';
             
end


Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;
             
% >>>>>>>>>>>>>>>>>>>>>>>
% Part 3 - Adds regularization
             
temp2 = Theta2;
temp2(:,1) = 0;
Theta2_grad = Theta2_grad + lambda * temp2 / m;

temp1 = Theta1;
temp1(:,1) = 0;
Theta1_grad = Theta1_grad + lambda * temp1 / m;

             
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
