function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

n = size(mu, 2);   % number of features
% fprintf('number of features: %d\n', n);

for i=1:n
    xi = X(:, i);  % extract feature 'i' from X
    mu(1, i) = mean(xi);
    sigma(1, i) = std(xi);
    X_norm(:,i) = (xi .- mu(1, i)) ./ sigma(1, i);
end;

% fprintf('X_norm = [%.4f], [%.4f]\n', X_norm');

% Print out 'mean' and 'std' for each feature
% fprintf('mean = [%.4f], [%.4f], std = [%.4f], [%.4f]\n', [mu sigma]');








% ============================================================

end
