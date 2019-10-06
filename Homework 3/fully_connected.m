%% Homework 3, Fully connected network
% Author: Anna Carlsson
% Last updated: 2019-10-06

%% Code
clc, clear all

% Load and pre-process dataset
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(1);
mean = mean(xTrain, 2);
xTrain = xTrain - mean;
xValid = xValid - mean;
xTest = xTest - mean;

% Settings
learning_rate = 0.1;
M1 = 2;
M2 = 2;
max_epochs = 20;
batch_size = 100;

% Initialize weights and thresholds
input_size = 3072;
output_size = 10;
[weights, thresholds] = initialize([M1, M2], input_size, output_size);

% Train network
epoch = 1;

while (epoch <= max_epochs)
    
    % Shuffle observations in training set
    index = randperm(size(xTrain,2));
    xTrain = xTrain(index,:);
    tTrain = tTrain(index,:);
    
    % For each minibatch, do:
    
    
end



%% Definition of functions
function [weights, thresholds] = initialize(M, input_size, output_size)
    nbr_layers = nnz(M);
    nbr_neurons = M(M~=0);
    
    if nbr_layers == 0
        weights = {normrnd(0, 1/input_size, [output_size, input_size])};
        thresholds = {zeros(10,1)};
    end
    
    if nbr_layers == 1
        weights = {normrnd(0, 1/input_size, [nbr_neurons, input_size]), normrnd(0, 1/nbr_neurons, [output_size, nbr_neurons])};
        thresholds = {zeros(nbr_neurons,1), zeros(10,1)};
    end
    
    if nbr_layers == 2
        weights = {normrnd(0, 1/input_size, [nbr_neurons(1), input_size]), normrnd(0, 1/nbr_neurons(1), [nbr_neurons(2), nbr_neurons(1)]), normrnd(0, 1/nbr_neurons(2), [output_size, nbr_neurons(2)])}; 
        thresholds = {zeros(nbr_neurons(1),1), zeros(nbr_neurons(2)), zeros(10,1)};
    end    
    
end