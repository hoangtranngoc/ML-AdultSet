train_file = 'C:\Users\E\OneDrive\matlabgit\ML-AdultSet\data\ohenc_data.train';
validation_file = 'C:\Users\E\OneDrive\matlabgit\ML-AdultSet\data\ohenc_data.val';
test_file = 'C:\Users\E\OneDrive\matlabgit\ML-AdultSet\data\ohenc_data.test';

%train_file = 'D:\E\Documents\machine learning - git\ML-AdultSet\data\ohenc_data.train';
%validation_file = 'D:\E\Documents\machine learning - git\ML-AdultSet\data\ohenc_data.validation';
%test_file = 'D:\E\Documents\machine learning - git\ML-AdultSet\data\ohenc_data.test';

delimiterIn = ' ';
headerlinesIn = 1;
TRAIN_SET = importdata(train_file, delimiterIn, headerlinesIn);
VALIDATION_SET = importdata(validation_file, delimiterIn, headerlinesIn);
TEST_SET = importdata(test_file, delimiterIn, headerlinesIn);

% Define the sets, and rotate the matrices

train_inputs = TRAIN_SET.data(:, 1:108);
train_outputs = TRAIN_SET.data(:, 109:110);
train_inputs = train_inputs';
train_outputs = train_outputs';

validation_inputs = VALIDATION_SET.data(:, 1:108);
validation_outputs = VALIDATION_SET.data(:, 109:110);
validation_inputs = validation_inputs';
validation_outputs = validation_outputs';

test_inputs = TEST_SET.data(:, 1:108);
test_outputs = TEST_SET.data(:, 109:110);
test_inputs = test_inputs';
test_outputs = test_outputs';

numberOfLayers=4;

% This piece of code increases the regularization value, step by step.
%for i=0.15:0.01:0.3 % Start at 0.1, increment by 0.1, end at 1
% for i=0.01:0.01:0.04
%     PerformTraining('trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, i);
% end

% ---- Initial display ----
% For the display
fprintf('regularizationvalue, learningrate, trainingsetMSE, trainingsetAccuracy, testsetAccurary, numberOfPerformedIterations, numberOfNeurons, numberOfLayers, sensitivity, specificity\n');
% For the file
fileID = fopen('results_trainscg_4layer_2to6neurons_1000epochs_testing.csv', 'w'); % Open file for writing
fprintf(fileID, 'regularizationvalue, learningrate, trainingsetMSE, trainingsetAccuracy, testsetAccurary, numberOfPerformedIterations, numberOfNeurons, numberOfLayers, sensitivity, specificity\n'); % Write first line to CSV-file

% Initialize the training
for numberOfNeurons=2:1:6 % Number of neurons. We assume that all of the layers have the same amount of neurons.
    for i=0.1:0.1:1 % Regularization value
        for j=0.000001:0.010000:0.1 % Learning rate
            PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, i, j, numberOfNeurons, numberOfLayers);
        end
    end
end

fclose(fileID);
    
%function result = PerformTraining(trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, numberOfEpochs, regularizationValue, learningRate)
function result = PerformTraining(fileID, trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, numberOfEpochs, regularizationValue, learningRate, numberOfNeurons, numberOfLayers)

    net = feedforwardnet( [ numberOfNeurons numberOfNeurons numberOfNeurons numberOfNeurons ], trainFunction); % Cut and paste numberOfNeurons to set the number of layers
    net.divideFcn = '';
        
    net = configure( net, train_inputs, train_outputs );

    %net.trainParam.showWindow = false; % Hide GUI
    
    net.trainParam.goal = 0.00001;
    net.trainParam.lr = learningRate;
    net.trainParam.show = 1; % Defines how often a status report shall be printed
    net.trainParam.epochs = numberOfEpochs;
    
    % Set the regularization value
    net.performParam.regularization = regularizationValue;
    

    net.trainParam.showCommandLine = false;

    [net,tr] = train( net, test_inputs, test_outputs );

    % Print the net
    %view(net)
    
    
    
    train_matrix = net(train_inputs);

    [ dummy , max_index_array] = max(train_matrix, [], 1);
    pred_outputs = max_index_array - 1;

    % calculate MSE
    perf = perform(net,train_matrix, train_outputs);

    nntraintool('close'); % Close the window
    
    % Generate a confusion matrix
    [c,cm,ind,per] = confusion(train_outputs, train_matrix);
        
    % --- Calculate specificity and sensitivity, using the confusion matrix ---
    
    % The matrix looks like this (source: https://se.mathworks.com/matlabcentral/answers/371767-confusion-matrix-results-sensitivity):
    % TN | FP
    % -------
    % FN | TP
    
    % Info from le Wikipedia: https://en.wikipedia.org/wiki/Confusion_matrix
    % The first element is 1,1, the second column on the first row is 1,2,
    % etc.
        
    % TP (sensitivity): When it's actually yes, how often does it predict yes?
    % The formula is: (TP)/(TP+FN)
    sensitivity=(cm(2,2))/(cm(2,2)+cm(2,1));
    
    % Specificity: When it's actually no, how often does it predict no?
    % The formula is: (TN)/(TN+FP)
    specificity = cm(1,1)/(cm(1,1)+cm(1,2));
    
    % Print the results
%     fprintf('----------------------------------------------------\n');
%     fprintf('Regularization value: %f\n', regularizationValue);
%     fprintf('Learning rate: %f\n', learningRate);
% 
%     fprintf('Training Set MSE: %f\n', perf);
%     fprintf('Training Set Accuracy: %f\n', mean(double(pred_outputs == train_outputs(2,:))) * 100);

    % Calculate accuracy on the validation set
    test_pred_matrix = net(test_inputs);
    [ dummy , test_max_index_array] = max(test_pred_matrix, [], 1);
    test_pred_outputs = test_max_index_array - 1;
    %fprintf('Validation Set Accuracy: %f\n', mean(double(validation_pred_outputs == validation_outputs(2,:))) * 100);
    
    %fprintf('Number of iterations (epochs) performed: %f\n', tr.num_epochs);
    
    % Generate CSV-style printout. This makes it much easier to import the
    % results into a spreadsheet, and discover the best values.

    trainsetAccuracy = mean(double(pred_outputs == train_outputs(2,:))) * 100;
    testsetAccuracy = mean(double(test_pred_outputs == test_outputs(2,:))) * 100;
    
    % Just a reminder for myself:
    %fprintf('regularizationvalue, learningrate, trainingsetMSE, trainingsetAccuracy, testsetAccurary, numberOfPerformedIterations, numberOfNeurons, numberOfLayers, sensitivity, specificity\n');
    fprintf('%f, %f, %f, %f, %f, %d, %d, %d, %f, %f\n', regularizationValue, learningRate, perf, trainsetAccuracy, testsetAccuracy, tr.num_epochs, numberOfNeurons, numberOfLayers, sensitivity, specificity);
    fprintf(fileID, '%f, %f, %f, %f, %f, %d, %d, %d, %f, %f\n', regularizationValue, learningRate, perf, trainsetAccuracy, testsetAccuracy, tr.num_epochs, numberOfNeurons, numberOfLayers, sensitivity, specificity); % To file
    
end
