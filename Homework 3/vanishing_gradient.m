%% Homework 3, Vanishing gradient
% Author: Anna Carlsson
% Last updated: 2019-10-08

%% Code

% Load and pre-process dataset
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(2);
mean = mean(xTrain, 2);
xTrain = xTrain - mean;
xValid = xValid - mean;
xTest = xTest - mean;

% Settings
learning_rate = 0.01;
nbr_layers = 5;
layer_size = 20;
max_epochs = 100;
batch_size = 100;

% Initialize weights and thresholds
input_size = 3072;
output_size = 10;
n = size(xTrain,2);
[weights, thresholds] = initialize(nbr_layers, layer_size, input_size, output_size);

%% Definition of functions
function [weights, thresholds, dW, dT] = initialize(nbr_layers, layer_size, input_size, output_size)

% Input layer -> first hidden
weights(1) = {normrnd(0, sqrt(1/input_size), [layer_size, input_size])};
dW(1) = {zeros(layer_size, input_size)};

% For each of the hidden layers -> next hidden layer
for i = 2 : 4
   weights(i) = {normrnd(0, sqrt(1/layer_size), [layer_size, layer_size])}; 
   dW(i) = {zeros(layer_size, layer_size)};
   thresholds(i-1) = {zeros(layer_size,1)};
   dT(i-1) = {zeros(layer_size,1)};
end

weights(5) = {normrnd(0, sqrt(1/input_size), [output_size, input_size])};
dW(5) = {zeros(output_size, layer_size)};
thresholds(4) = {zeros(output_size,1)};
dT(4) = {zeros(output_size,1)};

end
