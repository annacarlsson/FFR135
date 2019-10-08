%% Homework 3, Fully connected network
% Author: Anna Carlsson
% Last updated: 2019-10-08

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
M1 = 0;
M2 = 0;
max_epochs = 20;
batch_size = 100;

% Initialize weights and thresholds
input_size = 3072;
output_size = 10;
n = size(xTrain,2);
[weights, thresholds] = initialize([M1, M2], input_size, output_size);

% Create cells to save the weights and thresholds (to reuse later)
weights_all = cell(1, max_epochs);
thresholds_all = cell(1, max_epochs);

% Train network
epoch = 1;
val_errors = zeros(1, max_epochs);
train_errors = zeros(1, max_epochs);

disp(['----- TRAINING NETWORK -----'])
    
while (epoch <= max_epochs)
    
    % Shuffle observations in training set
    index = randperm(n);
    xTrain = xTrain(:,index);
    tTrain = tTrain(:,index);
    
    % For each minibatch, do:
    nbr_batches = n/batch_size;
    
    for i = 1 : nbr_batches
        
        x = xTrain(:, ((i-1) * batch_size + 1) : batch_size * i);
        t = tTrain(:, ((i-1) * batch_size + 1) : batch_size * i);
        
        n_layer = length(weights);
        
        % For each batch, reset gradients
        [dW, dTheta] = setup_gradient_arrays([M1, M2], input_size, output_size);
        
        % For each pattern in batch, do
        for j = 1 : batch_size
            state = cell(n_layer + 1, 1);
            x_batch = x(:,j);
            t_batch = t(:,j);
            V_temp = x_batch;
            state(1) = {V_temp};
            
            % Forward propagation
            for k = 1 : n_layer
                V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
                state(k+1) = {V_temp};
            end
            
            % Backpropagation
            delta = (t_batch - cell2mat(state(end))) .* g_prim(b(cell2mat(weights(end)), cell2mat(state(end-1)), cell2mat(thresholds(end))));
            dTheta(end) = {cell2mat(dTheta(end)) + delta};
            if size(cell2mat(dTheta(end)),2) == 2
                disp('hej')
            end
            dW(end) = {cell2mat(dW(end)) + delta * cell2mat(state(end-1))'};
            
            for k = 1 : (n_layer - 1)
                delta = (cell2mat(weights(end-k+1))' * delta) .* g_prim(b(cell2mat(weights(end-k)), cell2mat(state(end-k-1)), cell2mat(thresholds(end-k))));
                dTheta(end-k) = {cell2mat(dTheta(end-k)) + delta};
                dW(end-k) = {cell2mat(dW(end-k)) + delta * cell2mat(state(end-k-1))'};
            end
            
        end
        
        for j = 1 : n_layer
            weights(j) = {cell2mat(weights(j)) + learning_rate * cell2mat(dW(j))};
            thresholds(j) = {cell2mat(thresholds(j)) - learning_rate * cell2mat(dTheta(j))};
        end
        
    end
    
    % Compute error for validation set and training set
    state_val = cell(n_layer + 1, 1);
    m = size(xValid,2);
    
    % For each pattern in validation set
    outputs_val = zeros(10, m);
    for j = 1 : m
        x_val = xValid(:,j);
        V_temp = x_val;
            
        % Forward propagation
        for k = 1 : n_layer
            V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
        end
        outputs_val(:,j) = V_temp;
    end
    
    % Compute validation error
    temp_valerror = classification_error(tValid, outputs_val);
    val_errors(epoch) = temp_valerror;
    
    % For each pattern in training set
    outputs_train = zeros(10, n);
    for j = 1 : n
        x_train = xTrain(:,j);
        V_temp = x_train;
            
        % Forward propagation
        for k = 1 : n_layer
            V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
        end
        outputs_train(:,j) = V_temp;
    end
    
    % Compute validation error
    temp_trainerror = classification_error(tTrain, outputs_train);
    train_errors(epoch) = temp_trainerror;

    disp(['Epoch: ', num2str(epoch), '  Train error: ', num2str(temp_trainerror), ' Val. error: ', num2str(temp_valerror)])
    
    % Save weights and thresholds for current epoch
    weights_all(epoch) = weights;
    thresholds_all(epoch) = thresholds;
    
    epoch = epoch + 1;

end

%% Test network on test set
% Select the best model
[val, mod_index] = min(val_errors);
weights = weights_all(mod_index);
thresholds = thresholds_all(mod_index);

% Use this model to compute test error
outputs_test = zeros(10, size(xTest,2));
for j = 1 : size(xTest,2)
    x_test = xTest(:,j);
    V_temp = x_test;
    
    % Forward propagation
    for k = 1 : n_layer
        V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
    end
    outputs_train(:,j) = V_temp;
end

% Compute validation error
test_error = classification_error(tTest, outputs_test);

%% Definition of functions
function [weights, thresholds, dW, dT] = initialize(M, input_size, output_size)
nbr_layers = nnz(M);
nbr_neurons = M(M~=0);

if nbr_layers == 0
    weights = {normrnd(0, sqrt(1/input_size), [output_size, input_size])};
    thresholds = {zeros(10,1)};
    
    dW = {zeros(output_size, input_size)};
    dT = {zeros(10,1)};
end

if nbr_layers == 1
    weights = {normrnd(0, sqrt(1/input_size), [nbr_neurons, input_size]), normrnd(0, sqrt(1/nbr_neurons), [output_size, nbr_neurons])};
    thresholds = {zeros(nbr_neurons,1), zeros(10,1)};
    
    dW = {zeros(nbr_neurons, input_size), zeros(output_size, nbr_neurons)};
    dT = {zeros(nbr_neurons,1), zeros(10,1)};
end

if nbr_layers == 2
    weights = {normrnd(0, sqrt(1/input_size), [nbr_neurons(1), input_size]), normrnd(0, sqrt(1/nbr_neurons(1)), [nbr_neurons(2), nbr_neurons(1)]), normrnd(0, sqrt(1/nbr_neurons(2)), [output_size, nbr_neurons(2)])};
    thresholds = {zeros(nbr_neurons(1),1), zeros(nbr_neurons(2),1), zeros(10,1)};
    
    dW = {zeros(nbr_neurons(1), input_size), zeros(nbr_neurons(2), nbr_neurons(1)), zeros(output_size, nbr_neurons(2))};
    dT = {zeros(nbr_neurons(1),1), zeros(nbr_neurons(2),1), zeros(10,1)};
end

end

function [dW, dT] = setup_gradient_arrays(M, input_size, output_size)
nbr_layers = nnz(M);
nbr_neurons = M(M~=0);

if nbr_layers == 0
    dW = {zeros(output_size, input_size)};
    dT = {zeros(10,1)};
end

if nbr_layers == 1
    dW = {zeros(nbr_neurons, input_size), zeros(output_size, nbr_neurons)};
    dT = {zeros(nbr_neurons,1), zeros(10,1)};
end

if nbr_layers == 2
    dW = {zeros(nbr_neurons(1), input_size), zeros(nbr_neurons(2), nbr_neurons(1)), zeros(output_size, nbr_neurons(2))};
    dT = {zeros(nbr_neurons(1),1), zeros(nbr_neurons(2),1), zeros(10,1)};
end

end

function loc_field = b(weights, input, threshold)
loc_field = weights * input - threshold;
end

function sigm = sigmoid(val)
sigm = 1 ./ (1 + exp(-val));
end

function g = g_prim(bval)
g = sigmoid(bval) .* (1 - sigmoid(bval));
end

function C = classification_error(target, outputs)
length_valset = size(target, 2);    
C = 1 / (2*length_valset) * sum(abs(outputs - target),'all');
end