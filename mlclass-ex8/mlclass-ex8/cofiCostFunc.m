function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


[m n] = size(R);


%for i=1:m
%    for j=1:n
%        J = J + R(i,j) * ((Theta(j,:) * X(i,:)' - Y(i,j))^2 / 2);
%    end
%end


temp_J = (X * Theta' - Y).^2;
J = J + sum(sum(R.*temp_J)) / 2;

% Add regularization to cost function J.
J = J + lambda * sum(sum(Theta .^ 2)) / 2 + lambda * sum(sum(X .^ 2)) / 2;

          
% Compute X_grad.
          
%for i=1:m
%    for k=1:num_features
%        for j=1:n
%            X_grad(i,k) = X_grad(i,k) + R(i,j) * ((Theta(j,:) * X(i,:)' - Y(i,j)) * Theta(j,k));
%            % add regularization
%        end
%        X_grad(i,k) = X_grad(i,k) + lambda * X(i,k);
%    end
%end

for i=1:m
    % Get all the users that rated 'i' movie.
    idx = find(R(i, :)==1); % get index of users rated 'i' movie.
    Theta_temp = Theta(idx,:);
    Y_temp = Y(i, idx);
    X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp;

    % add regularization
    X_grad(i,:) = X_grad(i,:) + lambda * X(i, :);
end

                                           
                                           
% Compute Theta_grad.
for j=1:n
    for k=1:num_features
        for i=1:m
            Theta_grad(j,k) = Theta_grad(j,k) + R(i,j) * ((Theta(j,:) * X(i,:)' - Y(i,j)) * X(i,k));
            % add regularization
        end
        Theta_grad(j,k) = Theta_grad(j,k) + lambda * Theta(j,k);
    end
end


%for j=1:n
%    % Get all the movies that were rated by user 'j'.
%    idx = find(R(:, j)==1); % get index of movies
%    Theta_temp = Theta(idx,:);
%    Y_temp = Y(i, idx);
%    Theta_grad(j,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp;
%end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
