%% Homework 3, ReLU, softmax, early stopping
% Author: Anna Carlsson
% Last updated: 2019-10-09

%% Network 1
clc, clear all

% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

% Settings
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',400, ...
    'MiniBatchSize',8192, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',{xValid,tValid}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Plots','training-progress');

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
net_1 = trainNetwork(xTrain, tTrain, layers, options);

% Compute scores
[pred_train_1, scores_train_1] = classify(net_1, xTrain);
[pred_valid_1, scores_valid_1] = classify(net_1, xValid);
[pred_test_1, scores_test_1] = classify(net_1, xTest);

C_train_1 = classification_error(tTrain, pred_train_1);
C_valid_1 = classification_error(tValid, pred_valid_1);
C_test_1 = classification_error(tTest, pred_test_1);

%% Network 2
clc, clear all

% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

% Settings
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
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
net_2 = trainNetwork(xTrain, tTrain, layers, options);

% Compute scores
[pred_train_2, scores_train_2] = classify(net_2, xTrain);
[pred_valid_2, scores_valid_2] = classify(net_2, xValid);
[pred_test_2, scores_test_2] = classify(net_2, xTest);

C_train_2 = classification_error(tTrain, pred_train_2);
C_valid_2 = classification_error(tValid, pred_valid_2);
C_test_2 = classification_error(tTest, pred_test_2);

%% Network 3
clc, clear all

% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

% Settings
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',400, ...
    'MiniBatchSize',8192, ...
    'L2Regularization', 0.2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',{xValid,tValid}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Plots','training-progress');

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
net_3 = trainNetwork(xTrain, tTrain, layers, options);

% Compute scores
[pred_train_3, scores_train_2] = classify(net_3, xTrain);
[pred_valid_3, scores_valid_3] = classify(net_3, xValid);
[pred_test_3, scores_test_3] = classify(net_3, xTest);

C_train_3 = classification_error(tTrain, pred_train_3);
C_valid_3 = classification_error(tValid, pred_valid_3);
C_test_3 = classification_error(tTest, pred_test_3);

%% Definition of functions
function C = classification_error(target, outputs)
length_valset = size(target,1);    
C = 1 / length_valset * sum(outputs ~= target);
end
