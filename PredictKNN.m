function [label_test_pred] = PredictKNN(feature_train, label_train, feature_test, k)        
    % Input: feature_train - A matrix containing features of training
    %                        images. Dim: n_train x f
    %        label_train - An array containing the training labels for the
    %                      corresponding training features.
    %                      Dim: n_train x 1
    %        feature_test - A matrix containing features of testing images
    %                       Dim: n_test x f
    %        k - Value of k for knnsearch function
    % 
    % Output: label_test_pred - Predicted labels for feature_test.
    %                           Dim: n_test x 1
    %
    % Description: Runs KNN search for predicting labels for test data

    % Get matrix for knn indexes. Dim: n_test x k
    knn = knnsearch(feature_train, feature_test, 'K', k);
    
    % Get labels for corresponding knn indexes in knn matrix
    knn_labels = label_train(knn);  % Dim: n_test x k
    
    % Select most frequently occuring labels for each sample
    label_test_pred = mode(knn_labels, 2); % Dim: n_test x 1
end