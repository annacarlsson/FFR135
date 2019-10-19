%% Homework 3, Vanishing gradient
% Author: Anna Carlsson
% Last updated: 2019-10-17

%% Code
clc, clear all

% Load and pre-process dataset
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(2);
mean = mean(xTrain, 2);
xTrain = xTrain - mean;

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

% Compute initial learning speed
epoch = 1;
dT_all = cell(1, max_epochs + 1);

[dW, dTheta] = setup_gradient_arrays(layer_size, input_size, output_size);

for j = 1 : size(xTrain,2)
    state_init = cell(nbr_layers + 1, 1);
    x = xTrain(:,j);
    t = tTrain(:,j);
    V_temp = x;
    state_init(1) = {V_temp};
    
    % Forward propagation
    for k = 1 : nbr_layers
        V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
        state_init(k+1) = {V_temp};
    end
    
    % Backpropagation
    delta_ls = (t - cell2mat(state_init(end))) .* g_prim(b(cell2mat(weights(end)), cell2mat(state_init(end-1)), cell2mat(thresholds(end))));
    dTheta(end) = {cell2mat(dTheta(end)) + delta_ls};
    
    for k = 1 : (nbr_layers - 1)
        delta_ls = (cell2mat(weights(end-k+1))' * delta_ls) .* g_prim(b(cell2mat(weights(end-k)), cell2mat(state_init(end-k-1)), cell2mat(thresholds(end-k))));
        dTheta(end-k) = {cell2mat(dTheta(end-k)) + delta_ls};
    end
    
end

for layer = 1 : (nbr_layers)
    dTheta{layer} = dTheta{layer}/size(xTrain,2);
end

dT_all{epoch} = dTheta;

% Train network

disp(['----- TRAINING NETWORK -----'])

while (epoch <= max_epochs)
    
    disp(['Epoch: ', num2str(epoch)])
    
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
        [dW, dTheta] = setup_gradient_arrays(layer_size, input_size, output_size);
        
        % For each pattern in batch, do
        for j = 1 : batch_size
            state = cell(nbr_layers + 1, 1);
            x_batch = x(:,j);
            t_batch = t(:,j);
            V_temp = x_batch;
            state(1) = {V_temp};
            
            % Forward propagation
            for k = 1 : nbr_layers
                V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
                state(k+1) = {V_temp};
            end
            
            % Backpropagation
            delta = (t_batch - cell2mat(state(end))) .* g_prim(b(cell2mat(weights(end)), cell2mat(state(end-1)), cell2mat(thresholds(end))));
            dTheta(end) = {cell2mat(dTheta(end)) + delta};
            dW(end) = {cell2mat(dW(end)) + delta * cell2mat(state(end-1))'};
            
            for k = 1 : (nbr_layers - 1)
                delta = (cell2mat(weights(end-k+1))' * delta) .* g_prim(b(cell2mat(weights(end-k)), cell2mat(state(end-k-1)), cell2mat(thresholds(end-k))));
                dTheta(end-k) = {cell2mat(dTheta(end-k)) + delta};
                dW(end-k) = {cell2mat(dW(end-k)) + delta * cell2mat(state(end-k-1))'};
            end
            
        end
        
        for j = 1 : nbr_layers
            weights(j) = {cell2mat(weights(j)) + learning_rate * cell2mat(dW(j))};
            thresholds(j) = {cell2mat(thresholds(j)) - learning_rate * cell2mat(dTheta(j))};
        end
        
    end
    
    % Compute dT for all layers using all training data
    [dW, dTheta] = setup_gradient_arrays(layer_size, input_size, output_size);
    
    for j = 1 : size(xTrain,2)
        state = cell(nbr_layers + 1, 1);
        x = xTrain(:,j);
        t = tTrain(:,j);
        V_temp = x;
        state(1) = {V_temp};
        
        % Forward propagation
        for k = 1 : nbr_layers
            V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
            state(k+1) = {V_temp};
        end
        
        % Backpropagation
        delta_ls = (t - cell2mat(state(end))) .* g_prim(b(cell2mat(weights(end)), cell2mat(state(end-1)), cell2mat(thresholds(end))));
        dTheta(end) = {cell2mat(dTheta(end)) + delta_ls};
        
        for k = 1 : (nbr_layers - 1)
            delta_ls = (cell2mat(weights(end-k+1))' * delta_ls) .* g_prim(b(cell2mat(weights(end-k)), cell2mat(state(end-k-1)), cell2mat(thresholds(end-k))));
            dTheta(end-k) = {cell2mat(dTheta(end-k)) + delta_ls};
        end
        
    end
    
    for layer = 1 : (nbr_layers)
        dTheta{layer} = dTheta{layer}/size(xTrain,2);
    end
    
    dT_all{epoch + 1} = dTheta;
    epoch = epoch + 1;
end

%% Create plot of learning speeds

learning_speeds = zeros(max_epochs + 1, 5);

for e = 1 : (max_epochs + 1)
    for l = 1 : nbr_layers
        temp_U_vec = dT_all{e}{l};
        learning_speeds(e,l) = norm(temp_U_vec);
    end
end

epochs = linspace(1,max_epochs + 1,max_epochs + 1);
semilogy(epochs, learning_speeds(:,1),'LineWidth',1.5,'Color',[0, 0.4470, 0.7410]); hold on
semilogy(epochs, learning_speeds(:,2),'LineWidth',1.5,'Color',[0.4660, 0.6740, 0.1880]);
semilogy(epochs, learning_speeds(:,3),'LineWidth',1.5,'Color',[0.6350, 0.0780, 0.1840]);
semilogy(epochs, learning_speeds(:,4),'LineWidth',1.5,'Color',[0.9290, 0.6940, 0.1250]);
semilogy(epochs, learning_speeds(:,5),'LineWidth',1.5,'Color',[0.8500, 0.3250, 0.0980]);

ax = gca;
ax.FontSize = 11;

legend('Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Location', 'Southeast', 'fontsize',11);

title('Learning speeds for layers as function of epoch', 'fontsize',14)
xlabel('Epoch', 'fontsize', 12)
ylabel('Learning speed', 'fontsize',12)

%% Definition of functions
function [weights, thresholds] = initialize(layer_size, input_size, output_size)

% Input layer -> first hidden
weights(1) = {normrnd(0, sqrt(1/input_size), [layer_size, input_size])};

% For each of the hidden layers -> next hidden layer
for i = 2 : 4
    weights(i) = {normrnd(0, sqrt(1/layer_size), [layer_size, layer_size])};
    thresholds(i-1) = {zeros(layer_size,1)};
end

weights(5) = {normrnd(0, sqrt(1/input_size), [output_size, layer_size])};
thresholds(4) = {zeros(layer_size,1)};
thresholds(5) = {zeros(output_size,1)};
end

function [dW, dT] = setup_gradient_arrays(layer_size, input_size, output_size)
dW(1) = {zeros(layer_size, input_size)};

% For each of the hidden layers -> next hidden layer
for i = 2 : 4
    dW(i) = {zeros(layer_size, layer_size)};
    dT(i-1) = {zeros(layer_size,1)};
end

dW(5) = {zeros(output_size, layer_size)};
dT(4) = {zeros(layer_size,1)};
dT(5) = {zeros(output_size,1)};

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