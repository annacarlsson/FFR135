%% Homework 2, Two-layer perceptron
% Author: Anna Carlsson
% Last updated: 2019-09-29

%% Code
clc, clear all

% Load input data and define training, validation and target arrays
train = readtable('training_set.csv');
train = train{:, :};
val = readtable('validation_set.csv');
val = val{:, :};

% Settings
learning_rate = 0.02;
M1 = 3;
M2 = 3;
max_iter = 10000;

% Weight and threshold initialization 
W_1 = normrnd(0, 1, [2, M1]);
W_2 = normrnd(0, 1, [M1, M2]);
W_3 = normrnd(0, 1, [M2, 1]);

theta_1 = zeros(2);
theta_2 = zeros(M1);
theta_3 = zeros(M2);

% Train network
training_done = false;
i = 0;

while (i < max_iter && ~training_done)
    
    % Forward propagation
    
    % Backpropagation
    
    i = i + 1;
end


