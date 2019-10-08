%% Homework 3, Vanishing gradient
% Author: Anna Carlsson
% Last updated: 2019-10-08

%% Code
clc, clear all

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
[weights, thresholds] = initialize(layer_size, input_size, output_size);

% Train network
epoch = 1;

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
        
        % For each batch, reset gradients
        [dW, dTheta] = setup_gradient_arrays(nbr_layers, output_size);
        
        % For each pattern in batch, do
        for j = 1 : batch_size
            state = cell(nbr_layer + 1, 1);
            x_batch = x(:,j);
            t_batch = t(:,j);
            V_temp = x_batch;
            state(1) = {V_temp};
            
            % Forward propagation
            for k = 1 : nbr_layer
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
            
            for k = 1 : (nbr_layer - 1)
                delta = (cell2mat(weights(end-k+1))' * delta) .* g_prim(b(cell2mat(weights(end-k)), cell2mat(state(end-k-1)), cell2mat(thresholds(end-k))));
                dTheta(end-k) = {cell2mat(dTheta(end-k)) + delta};
                dW(end-k) = {cell2mat(dW(end-k)) + delta * cell2mat(state(end-k-1))'};
            end
            
        end
        
        for j = 1 : nbr_layer
            weights(j) = {cell2mat(weights(j)) + learning_rate * cell2mat(dW(j))};
            thresholds(j) = {cell2mat(thresholds(j)) - learning_rate * cell2mat(dTheta(j))};
        end
        
    end
    
    % Compute error for validation set and training set
    state_val = cell(nbr_layer + 1, 1);
    m = size(xValid,2);
    
    % For each pattern in validation set
    outputs_val = zeros(10, m);
    for j = 1 : m
        x_val = xValid(:,j);
        V_temp = x_val;
            
        % Forward propagation
        for k = 1 : nbr_layer
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
        for k = 1 : nbr_layer
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

%% Definition of functions
function [weights, thresholds] = initialize(layer_size, input_size, output_size)

% Input layer -> first hidden
weights(1) = {normrnd(0, sqrt(1/input_size), [layer_size, input_size])};

% For each of the hidden layers -> next hidden layer
for i = 2 : 4
   weights(i) = {normrnd(0, sqrt(1/layer_size), [layer_size, layer_size])}; 
   thresholds(i-1) = {zeros(layer_size,1)};
end

weights(5) = {normrnd(0, sqrt(1/input_size), [output_size, input_size])};
thresholds(4) = {zeros(output_size,1)};

end

function [dW, dT] = setup_gradient_arrays(layer_size, output_size)

% For each of the hidden layers -> next hidden layer
for i = 2 : 4
   dW(i) = {zeros(layer_size, layer_size)};
   dT(i-1) = {zeros(layer_size,1)};
end

dW(5) = {zeros(output_size, layer_size)};
dT(4) = {zeros(output_size,1)};

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
