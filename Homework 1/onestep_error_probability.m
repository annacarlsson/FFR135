%% Homework 1, One-step error probability
% Author: Anna Carlsson
% Last updated: 2019-09-14

%% Code - with diagonal elements wii set to zero
clc, clear all

% Generate patterns
nbr_patterns = [12, 24, 48, 70, 100, 120]';
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
        p = 2 * randi([0, 1], [N, nbr_p]) - 1;
        
        % Randomly select pattern to input and neuron i to update
        p_in = randi(nbr_p);
        n_i = randi(N);
        
        % Store pattern in network
        W_i = 1/N * p(n_i,:) * p';
        W_i(n_i) = 0;
        
        % Update neuron
        S0 = sign(p(n_i,p_in));
        S1 = sign(W_i * p(:,p_in)); 
        
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

%% Code - with diagonal elements wii NOT set to zero
clc, clear all

% Generate patterns
nbr_patterns = [12, 24, 48, 70, 100, 120]';
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
        p = 2 * randi([0, 1], [N, nbr_p]) - 1;
        
        % Randomly select pattern to input and neuron i to update
        p_in = randi(nbr_p);
        n_i = randi(N);
        
        % Store pattern in network
        W_i = 1/N * p(n_i,:) * p';
        %W_i(n_i) = 0;
        
        % Update neuron
        S0 = sign(p(n_i,p_in));
        S1 = sign(W_i * p(:,p_in)); 
        
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