%================================================%
% Prepare data
%================================================%
%train_file = '/home/e/OneDrive/matlabgit/ML-AdultSet/data/ohenc_data.train';
%test_file = '/home/e/OneDrive/matlabgit/ML-AdultSet/data/ohenc_data.test';

train_file = 'C:\Users\Blanco\OneDrive\matlabgit\ML-AdultSet\data\ohenc_data.train';
validation_file = 'C:\Users\Blanco\OneDrive\matlabgit\ML-AdultSet\data\ohenc_data.validation';
test_file = 'C:\Users\Blanco\OneDrive\matlabgit\ML-AdultSet\data\ohenc_data.test';

%train_file = 'D:\E\Documents\machine learning - git\ML-AdultSet\data\ohenc_data.train';
%validation_file = 'D:\E\Documents\machine learning - git\ML-AdultSet\data\ohenc_data.validation';
%test_file = 'D:\E\Documents\machine learning - git\ML-AdultSet\data\ohenc_data.test';

delimiterIn = ' ';
headerlinesIn = 1;
TRAIN_SET = importdata(train_file, delimiterIn, headerlinesIn);
VALIDATION_SET = importdata(validation_file, delimiterIn, headerlinesIn);
TEST_SET = importdata(test_file, delimiterIn, headerlinesIn);

% Define the training, validation and testing inputs, and 'rotate' them
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


% Define training functions, etc. here
trainFunction = 'trainlm';
PerformTraining(trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000);

% trainFunction = 'trainscg';
% PerformTraining(trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000);
% 
% trainFunction = 'trainbr';
% PerformTraining(trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, 1000);


% The function has a somewhat stupid name, to ensure that MATLAB does not
% confuse it with the ones included with the Neural Network Toolbox.

function result = PerformTraining(trainFunction, train_inputs, train_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs, numberOfEpochs)

    fprintf('----------------\nUsing the training function: %s\n', trainFunction); 
    
    %================================================%
    % Training network on training data set
    %================================================%
    net = feedforwardnet( [ 10 ], trainFunction); % May also specify the number of layers. For example 2, 5 will give two inputs, five layers

    % Configure the splits. The approach is "borrowed" from https://se.mathworks.com/matlabcentral/answers/115719-what-is-the-difference-between-divideblock-and-divideint
    net.divideFcn = 'divideblock';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.3;
    net.divideParam.testRatio = 0;
    
    %net.trainParam.showWindow = false; % Hide the annoying window.
    
    
    net = configure( net, train_inputs, train_outputs );
    %net.divideFcn = '';
    % net.trainParam.goal = 0.00001;
    net.trainParam.show = 1;
    net.trainParam.epochs = 1500;
    
    net.trainParam.showCommandLine = true;
    
    % Put custom training options here
    %trainingOptions('sgdm');
    %net.trainParam.epochs=50;
    
    [net,tr] = train( net, train_inputs, train_outputs );
    
    % Print the net
    view(net)
    pred_matrix = net(train_inputs);

    % each column contain predicted values for 2 nodes of output
    % get index of max values in a collumn
    [ dummy , max_index_array] = max(pred_matrix, [], 1);
    pred_outputs = max_index_array - 1;

    % calculate MSE
    perf = perform(net,pred_matrix, train_outputs);
    
    % compare pred_outputs to the last collumn, "outcome>50K", to calculate the
    % accuarcy
    fprintf('Training Set MSE: %f\n', perf);
    fprintf('Training Set Accuracy: %f\n', mean(double(pred_outputs == train_outputs(2,:))) * 100);

    % Calculate the performance of the test output
    
    
    %================================================%
    % Predicting result on validation set
    %================================================%
    % calclulate accuracy on test set
    validation_pred_matrix = net(validation_inputs);
    [ dummy , validation_max_index_array] = max(validation_pred_matrix, [], 1);
    validation_pred_outputs = validation_max_index_array - 1;
    fprintf('Validation Set Accuracy: %f\n', mean(double(validation_pred_outputs == validation_outputs(2,:))) * 100);

end

