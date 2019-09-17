%% Homework 1, Stochastic Hopfield network
% Author: Anna Carlsson
% Last updated: 2019-09-17

%% Code
clc, clear all

nbr_trails = 2*10^5;
N = 200; % number of neurons
p = 7; % number of random patterns

for i = 1:100
   
    % Generate p random patterns
    patterns = 2 * randi([0, 1], [N, p]) - 1;
    
    % Store patterns in network
    W = zeros(N, N);
    
    for k = 1 : p
        W = W + patterns(:, k) * patterns(:, k)';
    end
        
    W = W / N; % Normalize
    W = W - diag(diag(W)); % Set diagonal elements to zero
    
    S_0 = patterns(:,1);
    
    mu = zeros(1,nbr_trails);
    
    for j = 1 : nbr_trails
        S_1 = S_0;
        n_i = randi(N); % chose neuron randomly
        b_i = 1/N * W(n_i,:) * S_0;
        prob = sigmf(b_i, [4,0]);
        S_1(n_i) = randsrc(1, 1, [-1,1; prob, 1-prob]);
        mu(j) = 1/N * S_1' * patterns(:,1);
        S_0 = S_1;
    end    
    
end
mu_avg = 1/100 * sum(mu);
