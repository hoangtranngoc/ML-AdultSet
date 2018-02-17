%================================================%
% Prepare data
%================================================%
train_file = 'D:\Study\Ostfold\MachineLearning\git\data\ohenc_data.train';
test_file = 'D:\Study\Ostfold\MachineLearning\git\data\ohenc_data.test';
delimiterIn = ' ';
headerlinesIn = 1;
TRAIN_SET = importdata(train_file, delimiterIn, headerlinesIn);
TEST_SET = importdata(test_file, delimiterIn, headerlinesIn);

train_inputs = TRAIN_SET.data(:, 1:108);
train_outputs = TRAIN_SET.data(:, 109:110);
train_inputs = train_inputs';
train_outputs = train_outputs';

test_inputs = TEST_SET.data(:, 1:108);
test_outputs = TEST_SET.data(:, 109:110);
test_inputs = test_inputs';
test_outputs = test_outputs';


%================================================%
% Training network on training data set
%================================================%
net = feedforwardnet( [ 10 ] );
net = configure( net, train_inputs, train_outputs );
%net.divideFcn = '';
% net.trainParam.goal = 0.00001;
% net.trainParam.show = 1;
% net.trainParam.epochs = 20;
net = train( net, train_inputs, train_outputs );

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


%================================================%
% Predicting result on test set
%================================================%
% calclulate accuracy on test set
test_pred_matrix = net(test_inputs);
[ dummy , test_max_index_array] = max(test_pred_matrix, [], 1);
test_pred_outputs = test_max_index_array - 1;
fprintf('Test Set Accuracy: %f\n', mean(double(test_pred_outputs == test_outputs(2,:))) * 100);

