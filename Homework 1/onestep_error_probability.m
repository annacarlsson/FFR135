%% Homework 1, One-step error probability
% Author: Anna Carlsson
% Last updated: 2019-09-11

%% Code - with diagonal elements wii set to zero
clc, clear all

% Generate patterns
nbr_patterns = [12, 24, 48, 70, 100, 120]';
prob = 0.5;
N = 120;
nbr_trails = 10^5;
error_probs = [];

% For all different number of patterns, do
for i = 1 : length(nbr_patterns)
    nbr_p = nbr_patterns(i);
    error_count = 0;
    
    % Perform 10^5 independent trials
    for j = 1 : nbr_trails
        
        % Generate random patterns and store in network
        W = zeros(120,120);
        p = 2 * binornd(1, prob, [N, nbr_p]) - 1;
        
        for k = 1 : nbr_p
            W = W + p(:, k) * p(:, k)';
        end
        
        W = W / N; % Normalize
        W = W - diag(diag(W)); % Set diagonal elements to zero
        
        % Randomly select pattern to input and neuron i to update
        p_in = p(:, randi(nbr_p));
        n_i = randi(N);
        
        % Update neuron
        S0 = sign(p_in(n_i)); 
        S1 = sign(W(n_i,:) * p_in); 
        
        if S1 == 0
            S1 = 1;
        end
        
        % Check dynamics and see if correct
        if S0 ~= S1
            error_count = error_count + 1; 
        end    
    end
    
    temp_error_prob = error_count/nbr_trails;
    error_probs = [error_probs temp_error_prob];
end

% Result: p = [0.0006, 0.0117, 0.0539, 0.0953, 0.1358, 0.1577]

%% Code - with diagonal elements wii NOT set to zero
clc, clear all

% Generate patterns
nbr_patterns = [12, 24, 48, 70, 100, 120]';
prob = 0.5;
N = 120;
nbr_trails = 10^5;
error_probs = [];

% For all different number of patterns, do
for i = 1 : length(nbr_patterns)
    nbr_p = nbr_patterns(i);
    error_count = 0;
    
    % Perform 10^5 independent trials
    for j = 1 : nbr_trails
        
        % Generate random patterns and store in network
        W = zeros(120,120);
        p = 2 * binornd(1, prob, [N, nbr_p]) - 1;
        
        for k = 1 : nbr_p
            W = W + p(:, k) * p(:, k)';
        end
        
        W = W / N; % Normalize
        
        % Randomly select pattern to input and neuron i to update
        p_in = p(:, randi(nbr_p));
        n_i = randi(N);
        
        % Update neuron
        S0 = sign(p_in(n_i)); 
        S1 = sign(W(n_i,:) * p_in); 
        
        if S1 == 0
            S1 = 1;
        end
        
        % Check dynamics and see if correct
        if S0 ~= S1
            error_count = error_count + 1; 
        end    
    end
    
    temp_error_prob = error_count/nbr_trails;
    error_probs = [error_probs temp_error_prob];
end

% Result: p = [0.0002, 0.0033, 0.0123, 0.0179, 0.0215, 0.0224]