function [label_test_pred] = PredictSVM(feature_train, label_train, feature_test)
    % Input: feature_train - A matrix containing features of training
    %                        images. Dim: n_train x f
    %        label_train - An array containing the training labels for the
    %                      corresponding training features.
    %                      Dim: n_train x 1
    %        feature_test - A matrix containing features of testing images
    %                       Dim: n_test x f
    % 
    % Output: label_test_pred - Predicted labels for feature_test.
    %                           Dim: n_test x 1
    %
    % Description: This functions performs multiclass classification to
    %              train the model using SVM and predicts the labels for
    %              test data.
    
    lambda = 1e-3 ; % Regularization parameter
    maxIter = 1e6; % Maximum number of iterations
    
    % Train 15 SVMs using one-vs-all approach for each class and get the
    % scores for all the testing data.
    
    score_matrix = zeros(size(feature_test, 1), 15); % Dim: n_train x 15
    for i = 1 : 15
        y = zeros(size(feature_train, 1), 1);
        y(label_train==i) = 1; y(label_train~=i) = -1;
        
        weights = zeros(size(feature_train, 1), 1);
        weights(label_train==i) = 1500/(2*100);
        weights(label_train~=i) = 1500/(2*1400);

        [w, b, ~] = vl_svmtrain(transpose(feature_train), transpose(y),...
            lambda, 'MaxNumIterations', maxIter, 'weights', weights,...
            'loss', 'HINGE2');
        
        scores = transpose(w)*transpose(feature_test) + b;
        score_matrix(:, i) = scores;
    end
    
    % Set appropriate class label for each testing sample using the SVM 
    % score with highest confidence.
    label_test_pred = zeros(size(feature_test, 1), 1);
    for i = 1 : size(feature_test, 1)
        max_score = max(score_matrix(i, :));
        label_test_pred(i) = find(score_matrix(i,:) == max_score);
    end
end