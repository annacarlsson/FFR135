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
n = length(train);
m = length(val);

% Settings
learning_rate = 0.02;
M1 = 3;
M2 = 3;
max_epochs = 10000;

% Weight and threshold initialization 
W_1 = normrnd(0, 1, [M1, 2]);
W_2 = normrnd(0, 1, [M2, M1]);
W_3 = normrnd(0, 1, [1, M2]);

theta_1 = zeros(2,1);
theta_2 = zeros(M1,1);
theta_3 = zeros(M2,1);

% Train network
training_done = false;
i = 0;
epoch = 1;

while (i < max_epochs * n && ~training_done)
    
    mu = randi(n);
    input_pattern = train(mu);
    
    % Forward propagation
    V1 = forward_propagation(theta_1, W_1, input_pattern);
    V2 = forward_propagation(theta_2, W_2, V1);
    output = forward_propagation(theta_3, W_3, V2);
    
    % Backpropagation
    
    % After each epoch, compute classification error from validation set
    if (mod(i, m) == 0)
        outputs = zeros(m,1);
        
        for j = 1 : m
            input_pattern_val = val(j,:);
            V1_val = forward_propagation(theta_1, W_1, input_pattern_val);
            V2_val = forward_propagation(theta_2, W_2, V1);
            outputs(j) = forward_propagation(theta_3, W_3, V2);
        end
        
        val_errors = classification_error(outputs, val, m);
        print(val_errors)
        
        if (val_errors < 0.12)
            training_done = true;
        end
            
        epoch = epoch + 1; 
    end
    
    i = i + 1;
    
end

% Definition of functions
function V = forward_propagation(theta, W, input)
    V = tanh(W * input - theta);
end

function C = classification_error(outputs, valset, length_valset)
    C = 1 / (2*length_valset) * sum(sign(outputs) - valset(:,3));
end
