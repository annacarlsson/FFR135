%% Homework 2, Linear separability of 4-dimensional Boolean functions
% Author: Anna Carlsson
% Last updated: 2019-10-06

%% Code
clc, clear all

% Load input data and convert to matrix
x = readtable('input_data_numeric.csv');
x = x{:, :};
x(:,1) = [];

% Targets
A = [1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1];
B = [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1];
C = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1];
D = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1];
E = [-1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1];
F = [1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1];

t = D; % choose which boolean function to use

% Settings
max_iter = 10^5;
learning_rate = 0.02;

% Initialize weights and threshold
N = size(x, 2); 
W = -0.2 + (0.2 + 0.2) * rand(N,1);
theta = -1 + 2 * rand(1);
output = NaN(size(x,1),1)';

for i = 1 : max_iter
    
    % Forward propagation
    mu = randi(16);
    x_mu = x(mu, :);
    t_mu = t(mu);
    b_mu = b(W, x_mu, theta);
    
    O_mu = O(b_mu);
    output(mu) = O_mu;
    
    if (isequal(sign(output), t))
        break
    end
    
    % Backpropagation
    dtheta = dHdtheta(learning_rate, b_mu, O_mu, t_mu);
    dw = dHdw(learning_rate, b_mu, O_mu, t_mu, x_mu);
    
    theta = theta + dtheta;
    W = W + dw;
    
end

%% Definition of functions
function b_mu = b(W, x_mu, theta)
    b_mu = -theta + W' * x_mu';
end

function O_mu = O(b_mu)
    O_mu = tanh(0.5 * b_mu);
end

function dtheta = dHdtheta(learning_rate, b_mu, O_mu, t_mu)
    dtheta = - 0.5 * learning_rate * (t_mu - O_mu) * (1 - b_mu^2);
end

function dw = dHdw(learning_rate, b_mu, O_mu, t_mu, x_mu)
    dw = (0.5 * learning_rate * (t_mu - O_mu) * (1 - b_mu^2) * x_mu)';
end