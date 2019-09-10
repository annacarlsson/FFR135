%% Homework 1, One-step error probability
% Author: Anna Carlsson
% Last updated: 2019-09-10

%% Problem
% Asynchronous deterministic updates for Hopfield network
% Hebb's rule with wii=0
% Estimate one-step error probability for each p

%% Code

% Generate patterns
nbr_patterns = [12,24,48,70,100,120]';
prob = 0.5;
N = 120;

for i = 1 : 2 %length(nbr_patterns)
    nbr_p = nbr_patterns(i);
    p = 2*binornd(1,prob,[N,1])-1;
end
