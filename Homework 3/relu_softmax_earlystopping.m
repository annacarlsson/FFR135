%% Homework 3, ReLU, softmax, early stopping
% Author: Anna Carlsson
% Last updated: 2019-10-09

%% Network 1
clc, clear all

% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

% Settings
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',400, ...
    'MiniBatchSize',8192, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',{xValid,tValid}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Plots','training-progress')

% Define layers
layers = [
    imageInputLayer([32 32 3])
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Train network
net = trainNetwork(xTrain, tTrain, layers, options);

% Compute scores
[pred_train, scores_train] = classify(net, xTrain);
[pred_valid, scores_valid] = classify(net, xValid);
[pred_test, scores_test] = classify(net, xTest);

C_train = classification_error(tTrain, pred_train);
C_valid = classification_error(tValid, pred_valid);
C_test = classification_error(tTest, pred_test);

%% Network 2
clc, clear all

% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

% Settings
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.003, ...
    'MaxEpochs',400, ...
    'MiniBatchSize', 8192, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',{xValid,tValid}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Plots','training-progress')

% Define layers
layers = [
    imageInputLayer([32 32 3])
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Train network
net = trainNetwork(xTrain, tTrain, layers, options);

% Compute scores
[pred_train, scores_train] = classify(net, xTrain);
[pred_valid, scores_valid] = classify(net, xValid);
[pred_test, scores_test] = classify(net, xTest);

C_train = classification_error(tTrain, pred_train);
C_valid = classification_error(tValid, pred_valid);
C_test = classification_error(tTest, pred_test);

%% Network 3
clc, clear all

% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

% Settings
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',400, ...
    'MiniBatchSize',8192, ...
    'L2Regularization', 0.2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',{xValid,tValid}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Plots','training-progress')

% Define layers
layers = [
    imageInputLayer([32 32 3])
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Train network
net = trainNetwork(xTrain, tTrain, layers, options);

% Compute scores
[pred_train, scores_train] = classify(net, xTrain);
[pred_valid, scores_valid] = classify(net, xValid);
[pred_test, scores_test] = classify(net, xTest);

C_train = classification_error(tTrain, pred_train);
C_valid = classification_error(tValid, pred_valid);
C_test = classification_error(tTest, pred_test);

%% Definition of functions
function C = classification_error(target, outputs)
length_valset = size(target,1);    
C = 1 / length_valset * sum(outputs ~= target);
end
