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


% ---- Initial display ----
% For the display
%fprintf('regularizationvalue, learningrate, trainingsetMSE, trainingsetAccuracy, testsetAccurary, numberOfPerformedIterations, numberOfNeurons, numberOfLayers, trainingsetMSE, trainingsetAccuracy, sensitivity, specificity\n');
fprintf('regularizationValue, learningRate, trainingsetMSE, trainingsetAccuracy, validationsetAccuracy, testsetAccuracy, numberOfPerformedIterationsTraining, numberOfEpochsTestSet, numberOfNeurons, numberOfLayers, sensitivity, specificity\n');
% For the file
fileID='none';
% Initialize the training, using the testing set.
% Template:
% fileID, trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, numberOfEpochs, regularizationValue, learningRate, trainingsetMSE, trainingsetAccuracy, validationsetAccuracy, numberOfPerformedIterationsTraining, numberOfNeurons, numberOfLayers)


% Set the parameters below. The values are taken from the CSV-files from
% earlier results. There's quite a bit of search and replace, and manual 
% editing to do, to produce the line(s) below.
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.080000,0.104070,84.324490,85.104423,10000,4,2);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.070000,0.104138,84.113334,84.843366,10000,6,2);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.100000,0.104502,83.998157,84.720516,10000,5,2);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.070000,0.104689,84.136369,84.628378,10000,3,2);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.090000,0.103814,84.240028,84.582310,10000,4,2);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.300000,0.060000,0.092795,84.109494,84.966216,10000,6,3);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.090000,0.108625,83.660306,84.889435,10000,6,3);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.100000,0.108341,83.721734,84.858722,10000,5,3);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.090000,0.106907,84.205475,84.843366,10000,5,3);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.060000,0.107286,84.086459,84.812654,10000,5,3);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.070000,0.112938,83.671824,85.150491,10000,6,4);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.200000,0.060000,0.106699,84.044228,85.119779,10000,6,4);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.200000,0.080000,0.102516,84.086459,84.950860,10000,5,4);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.300000,0.030000,0.106160,84.236188,84.874079,10000,6,4);
PerformTraining(fileID, 'traingd', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 10000, 0.100000,0.090000,0.111204,83.875302,84.858722,10000,6,4);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.040000,0.108397,84.001996,88.175676,1000,6,2);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.070000,0.105238,84.531808,88.114251,1000,6,2);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.050000,0.106604,84.289937,87.791769,1000,6,2);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.010000,0.106572,84.201636,87.714988,1000,6,2);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.200000,0.030000,0.094577,84.439667,87.714988,1000,6,2);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.030000,0.107149,84.140208,88.605651,1000,6,3);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.030000,0.105367,84.193957,88.191032,1000,5,3);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.300000,0.060000,0.084207,84.712251,88.114251,1000,6,3);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.040000,0.106495,84.190118,88.052826,1000,6,3);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.200000,0.100000,0.097361,84.186279,88.052826,1000,6,3);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.070000,0.110445,83.740930,88.897420,1000,6,4);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.010000,0.112102,83.890659,88.728501,1000,6,4);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.060000,0.107532,84.109494,88.559582,1000,6,4);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.020000,0.109656,83.975122,88.482801,1000,6,4);
PerformTraining(fileID, 'trainscg', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000, 0.100000,0.040000,0.107319,84.128690,88.467445,1000,6,4);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.030000,0.090347,84.942604,85.918305,56,5,2);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.100000,0.070000,0.097269,84.996353,85.181204,9,6,2);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.100000,0.090030,85.053941,85.073710,53,4,2);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.020000,0.089022,84.827427,84.889435,42,4,2);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.100000,0.090000,0.096711,84.942604,84.843366,11,5,2);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.100000,0.090000,0.099473,84.566361,86.655405,28,6,3);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.300000,0.050000,0.080472,84.934925,85.242629,146,4,3);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.040000,0.088615,84.934925,84.659091,9,5,3);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.500000,0.050000,0.059058,84.631627,84.643735,250,6,3);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.050000,0.086967,84.911890,84.613022,9,6,3);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.050000,0.096631,84.193957,87.684275,80,4,4);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.200000,0.090000,0.090765,84.831267,85.626536,12,4,4);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.100000,0.030000,0.101121,84.585557,85.519042,23,3,4);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.100000,0.070000,0.101325,84.766000,85.042998,7,5,4);
PerformTraining(fileID, 'trainlm', train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 250, 0.100000,0.100000,0.095896,84.888855,84.889435,12,6,4);



fclose(fileID);
    
function result = PerformTraining(fileID, trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, numberOfEpochs, regularizationValue, learningRate, trainingsetMSE, trainingsetAccuracy, validationsetAccuracy, numberOfPerformedIterationsTraining, numberOfNeurons, numberOfLayers)

    % How many layers do you want today?
    if numberOfLayers==2
        net = feedforwardnet( [ numberOfNeurons numberOfNeurons ], trainFunction);
    elseif numberOfLayers==3
        net = feedforwardnet( [ numberOfNeurons numberOfNeurons numberOfNeurons ], trainFunction); 
    elseif numberOfLayers==4
        net = feedforwardnet( [ numberOfNeurons numberOfNeurons numberOfNeurons numberOfNeurons ], trainFunction);
    end
    
    net.divideFcn = '';
        
    net = configure( net, train_inputs, train_outputs );

    %net.trainParam.showWindow = false; % Hide GUI

    % Set up some parameters
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
    
    % Print results to screen, and/or output to a file.
    fprintf("'%s',%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%f,%f\n", trainFunction, regularizationValue, learningRate, trainingsetMSE, trainingsetAccuracy, validationsetAccuracy, testsetAccuracy, numberOfPerformedIterationsTraining, tr.num_epochs, numberOfNeurons, numberOfLayers, sensitivity, specificity);
    % Uncomment the line below to save the results to a file. Remember to
    % set "fileID".
    %fprintf(fileID, "'%s',%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%f,%f\n", trainFunction, regularizationValue, learningRate, trainingsetMSE, trainingsetAccuracy, validationsetAccuracy, testsetAccuracy, numberOfPerformedIterationsTraining, tr.num_epochs, numberOfNeurons, numberOfLayers, sensitivity, specificity);

end
