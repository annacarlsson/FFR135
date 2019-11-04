%% Homework 3, Fully connected network
% Author: Anna Carlsson
% Last updated: 2019-10-15

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
M1 = 50;
M2 = 50;
max_epochs = 20;
batch_size = 100;

% Initialize weights and thresholds
input_size = 3072;
output_size = 10;
n = size(xTrain,2);
m = size(xValid,2);
[weights, thresholds] = initialize([M1, M2], input_size, output_size);

% Create cells to save the weights and thresholds (to reuse later)
weights_all = cell(1, max_epochs);
thresholds_all = cell(1, max_epochs);

% Train network
epoch = 1;
val_errors = zeros(1, max_epochs + 1);
train_errors = zeros(1, max_epochs + 1);

% Compute training and validation error before any training (for plot)
n_layer = length(weights);
state_val_init = cell(n_layer + 1, 1);

outputs_val_init = zeros(10, m);
for j = 1 : m
    x_val_init = xValid(:,j);
    V_temp = x_val_init;
    
    % Forward propagation
    for k = 1 : n_layer
        V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
    end
    outputs_val_init(:,j) = V_temp;
end

% Compute validation error
val_errors(1) = classification_error(tValid, outputs_val_init);

% For each pattern in training set
outputs_train_init = zeros(10, n);
for j = 1 : n
    x_train_init = xTrain(:,j);
    V_temp = x_train_init;
    
    % Forward propagation
    for k = 1 : n_layer
        V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
    end
    outputs_train_init(:,j) = V_temp;
end

% Compute validation error
train_errors(1) = classification_error(tTrain, outputs_train_init);

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
    val_errors(epoch + 1) = temp_valerror;
    
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
    train_errors(epoch + 1) = temp_trainerror;
    
    disp(['Epoch: ', num2str(epoch), '  Train error: ', num2str(temp_trainerror), ' Val. error: ', num2str(temp_valerror)])
    
    % Save weights and thresholds for current epoch
    weights_all{epoch} = weights;
    thresholds_all{epoch} = thresholds;
    
    epoch = epoch + 1;
    
end

%% Test network on test set
% Select the best model
[val, mod_index] = min(val_errors);
weights = weights_all{mod_index-1};
thresholds = thresholds_all{mod_index-1};

% Use this model to compute test error
outputs_test = zeros(10, size(xTest,2));
for j = 1 : size(xTest,2)
    x_test = xTest(:,j);
    V_temp = x_test;
    
    % Forward propagation
    for k = 1 : n_layer
        V_temp = sigmoid(b(cell2mat(weights(k)), V_temp, cell2mat(thresholds(k))));
    end
    outputs_test(:,j) = V_temp;
end

% Compute test error 
test_error = classification_error(tTest, outputs_test);

%% Plot validation and train errors
train_error_1 = [0.9050, 0.6722, 0.6390, 0.6257, 0.6353, 0.6109, 0.5974, 0.5971, 0.6149, 0.6014, 0.5976, 0.5878, 0.5920, 0.5945, 0.5929, 0.5867, 0.5785, 0.5794, 0.5801, 0.5747, 0.5904];
val_error_1 = [0.9012, 0.6895, 0.6620, 0.6485, 0.6595, 0.6464, 0.6389, 0.6288, 0.6514, 0.6423, 0.6389, 0.6305, 0.6335, 0.6401, 0.6409, 0.6361, 0.6289, 0.6268, 0.6328, 0.6247, 0.6383];

train_error_2 = [0.9006, 0.6091, 0.5886, 0.5791, 0.5719, 0.5748, 0.5651, 0.5707, 0.5674, 0.5487, 0.5480, 0.5484, 0.5474, 0.5433, 0.5391, 0.5329, 0.5348, 0.5327, 0.5316, 0.5389, 0.5272];
val_error_2 = [0.8978, 0.6284, 0.6185, 0.6114, 0.6070, 0.6130, 0.5967, 0.6090, 0.6135, 0.6002, 0.6011, 0.5970, 0.6028, 0.6021, 0.6006, 0.5955, 0.6041, 0.5998, 0.6006, 0.6043, 0.6004];

train_error_3 = [0.8954,0.8477,0.7462,0.6880,0.5924,0.5650,0.5482,0.5050,0.4888,0.4908,0.4660,0.4516,0.4478,0.4416,0.4327,0.4367,0.4228,0.4163,0.4128,0.4061,0.4073];
val_error_3 = [0.8974,0.8496,0.7572,0.7115,0.6233,0.6072,0.5951,0.5599,0.5552,0.5531,0.5432,0.5338,0.5482,0.5407,0.5482,0.5405,0.5411,0.5429,0.5425,0.5399,0.5453];

train_error_4 = [0.9006,0.8995,0.6654,0.5926,0.5609,0.5486,0.5215,0.5177,0.4982,0.4929,0.4876,0.4750,0.4721,0.4549,0.4429,0.4371,0.4317,0.4245,0.4299,0.4146,0.4175];
val_error_4 = [0.8975,0.9020,0.6735,0.6062,0.5900,0.5862,0.5667,0.5672,0.5540,0.5514,0.5581,0.5489,0.5522,0.5481,0.5438,0.5386,0.5431,0.5401,0.5491,0.5420,0.5469];

epochs = linspace(0,20,21);
semilogy(epochs, train_error_1,'LineWidth',1.5,'Color',[0, 0.4470, 0.7410]); hold on
semilogy(epochs, val_error_1,'--','LineWidth',1.5, 'Color',[0, 0.4470, 0.7410]);
semilogy(epochs, train_error_2,'LineWidth',1.5,'Color',[0.4660, 0.6740, 0.1880]); 
semilogy(epochs, val_error_2,'--','LineWidth',1.5, 'Color',[0.4660, 0.6740, 0.1880]);
semilogy(epochs, train_error_3,'LineWidth',1.5,'Color',[0.6350, 0.0780, 0.1840]); 
semilogy(epochs, val_error_3,'--','LineWidth',1.5, 'Color',[0.6350, 0.0780, 0.1840]);
semilogy(epochs, train_error_4,'LineWidth',1.5,'Color',[0.9290, 0.6940, 0.1250]); 
semilogy(epochs, val_error_4,'--','LineWidth',1.5, 'Color',[0.9290, 0.6940, 0.1250]);

ax = gca;
ax.FontSize = 11; 

legend('Net 1 train', 'Net 1 val', 'Net 2 train', 'Net 2 val', 'Net 3 train', 'Net 3 val', 'Net 4 train', 'Net 4 val', 'Location', 'Northeast', 'fontsize',11);

title('Training and validation errors per epoch', 'fontsize',14)
xlabel('Epoch', 'fontsize', 12)
ylabel('Classification error', 'fontsize',12)

%% Definition of functions
function [weights, thresholds] = initialize(M, input_size, output_size)
nbr_layers = nnz(M);
nbr_neurons = M(M~=0);

if nbr_layers == 0
    weights = {normrnd(0, sqrt(1/input_size), [output_size, input_size])};
    thresholds = {zeros(10,1)};
end

if nbr_layers == 1
    weights = {normrnd(0, sqrt(1/input_size), [nbr_neurons, input_size]), normrnd(0, sqrt(1/nbr_neurons), [output_size, nbr_neurons])};
    thresholds = {zeros(nbr_neurons,1), zeros(10,1)};
end

if nbr_layers == 2
    weights = {normrnd(0, sqrt(1/input_size), [nbr_neurons(1), input_size]), normrnd(0, sqrt(1/nbr_neurons(1)), [nbr_neurons(2), nbr_neurons(1)]), normrnd(0, sqrt(1/nbr_neurons(2)), [output_size, nbr_neurons(2)])};
    thresholds = {zeros(nbr_neurons(1),1), zeros(nbr_neurons(2),1), zeros(10,1)};
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
outputs = double(bsxfun(@eq, outputs, max(outputs, [], 1)));
C = 1 / (2*length_valset) * sum(abs(outputs - target),'all');
end