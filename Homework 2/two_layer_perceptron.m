%% Homework 2, Two-layer perceptron
% Author: Anna Carlsson
% Last updated: 2019-10-06

%% Code
clc, clear all

% Load input data and define training, validation and target arrays
train = readtable('training_set.csv');
train = train{:, :};
val = readtable('validation_set.csv');
val = val{:, :};
n = length(train);
m = length(val);

% Plot of dataset with classes
class_one = train(:,3) == 1;
figure(1)
plot(train(class_one, 1), train(class_one, 2), 'r.')
hold on
plot(train(~class_one, 1), train(~class_one, 2), 'b.')

% Settings
learning_rate = 0.01;
M1 = 10;
M2 = 5;
max_epochs = 1000;

% Weight and threshold initialization
W1 = normrnd(0, 1, [M1, 2]);
W2 = normrnd(0, 1, [M2, M1]);
W3 = normrnd(0, 1, [1, M2]);

theta1 = zeros(M1,1);
theta2 = zeros(M2,1);
theta3 = zeros(1,1);

% Train network
training_done = false;
epoch = 1;

val_errors = zeros(1, max_epochs);

while (epoch < max_epochs && ~training_done)
    
    % For one epoch, do
    for j = 1 : n
        mu = randi(n);
        x = train(mu, 1:2)';
        t = train(mu, 3);
        
        % Forward propagation
        V1 = tanh(b(theta1, W1, x));
        V2 = tanh(b(theta2, W2, V1));
        output = tanh(b(theta3, W3, V2));
        
        % Backpropagation
        delta = (t - output) .* g_prim(b(theta3, W3, V2));
        dTheta3 = delta;
        dW3 = delta .* V2';
        delta = delta .* W3' .* g_prim(b(theta2, W2, V1));
        dTheta2 = delta;
        dW2 = delta * V1';
        delta = (W2' * delta) .* g_prim(b(theta1, W1, x));
        dTheta1 = delta;
        dW1 = delta * x';
        
        W1 = W1 + learning_rate * dW1;
        W2 = W2 + learning_rate * dW2;
        W3 = W3 + learning_rate * dW3;

        theta1 = theta1 - learning_rate * dTheta1;
        theta2 = theta2 - learning_rate * dTheta2;
        theta3 = theta3 - learning_rate * dTheta3;
        
    end
    
    % After each epoch, compute classification error from validation set
    outputs = zeros(m,1);
    
    for k = 1 : m
        x_val = val(k,1:2)';
        V1_val = tanh(b(theta1, W1, x_val));
        V2_val = tanh(b(theta2, W2, V1_val));
        outputs(k) = tanh(b(theta3, W3, V2_val));
    end
    
    val_error = classification_error(outputs, val, m);
    val_errors(epoch) = val_error;
    disp(['Epoch: ', num2str(epoch), ' Validation error: ', num2str(val_error)])
    
    if (val_error < 0.12)
        training_done = true;
    end
    
    epoch = epoch + 1;
end

% Plot predictions of validation set
figure(2)
class_one = sign(outputs) == 1;
plot(val(class_one, 1), val(class_one, 2), 'r.')
hold on
plot(val(~class_one, 1), val(~class_one, 2), 'b.')

%% Export vectors
csvwrite('w1.csv',W1)
csvwrite('w2.csv',W2)
csvwrite('w3.csv',W3')

csvwrite('t1.csv',theta1)
csvwrite('t2.csv',theta2)
csvwrite('t3.csv',theta3)

%% Definition of functions
function C = classification_error(outputs, valset, length_valset)
C = 1 / (2*length_valset) * sum(abs(sign(outputs) - valset(:,3)));
end

function bval = b(theta, W, input)
    bval = W * input - theta;
end

function gprim = g_prim(bval)
    gprim = 1 - tanh(bval).^2;
end