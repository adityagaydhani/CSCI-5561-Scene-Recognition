function [confusion, accuracy] = ClassifyKNN_BoW
    %%
    % Get labels and paths for all images from text files
    file_name = './objects/fileData.mat';
    if isfile(file_name)
        load(file_name, 'label_train', 'label_test',...
            'paths_train', 'paths_test');
    else
        [label_train, paths_train] = getDataFromFile('./train.txt');
        [label_test, paths_test] = getDataFromFile('./test.txt');
        save(file_name, 'label_train', 'label_test',...
            'paths_train', 'paths_test');
    end
    n_train = size(label_train, 1); n_test = size(label_test, 1);
    
    %%
    % Generate image cell to store every training image
    file_name = './objects/imageData.mat';
    if isfile(file_name)
        load(file_name, 'training_image_cell');
    else
        training_image_cell = getImageCell(paths_train);
    end
    
    % Generate image cell to store every testing image
    if isfile(file_name)
        load(file_name, 'testing_image_cell');
    else
        testing_image_cell = getImageCell(paths_test);
    end
    
    save(file_name, 'training_image_cell', 'testing_image_cell');
    
    %%
    % Get dense SIFT features from training images
    file_name = './objects/denseSiftFeatures.mat';
    if isfile(file_name)
        load(file_name, 'features_train');
    else
        features_train = cell(n_train);
        for i = 1 : n_train
            I = training_image_cell{i};
            I = single(vl_imdown(I));
            [~, d] = vl_dsift(I, 'size', 8, 'step', 2, 'fast'); % 128 x p
            features_train{i} = transpose(d); % p x 128
        end
    end

    % Get dense SIFT features from testing images
    if isfile(file_name)
        load(file_name, 'features_test');
    else
        features_test = cell(n_test);
        for i = 1 : n_test
            I = testing_image_cell{i};
            I = single(vl_imdown(I));
            [~, d] = vl_dsift(I, 'size', 8, 'step', 2, 'fast'); % 128 x p
            features_test{i} = transpose(d); % p x 128
        end
    end
    
    save(file_name, 'features_train', 'features_test');
    
    %%
    % Build vocabulary
    dic_size = 50;
    file_name = './objects/vocab.mat';
    if isfile(file_name)
        load(file_name, 'vocab');
    else
        vocab = BuildVisualDictionary(training_image_cell, dic_size);
        save(file_name, 'vocab')
    end
    
    %%
    % Compute BoW training features
    file_name = './objects/bowFeatures.mat';
    if isfile(file_name)
        load(file_name, 'bow_features_train');
    else
        bow_features_train = zeros(n_train, dic_size);
        for i = 1 : n_train
            bow_features_train(i, :) = transpose(...
                ComputeBoW(features_train{i}, vocab));
        end
    end
    
    % Compute BoW testing features
    if isfile(file_name)
        load(file_name, 'bow_features_test');
    else
        bow_features_test = zeros(n_test, dic_size);
        for i = 1 : n_test
            bow_features_test(i, :) = transpose(...
                ComputeBoW(features_test{i}, vocab));
        end
    end
    
    save(file_name, 'bow_features_train', 'bow_features_test');
    
    %%
    % Predict labels for test images
    label_test_pred = PredictKNN(bow_features_train, label_train,...
        bow_features_test, 10);
    
    % Compute confusion matrix and accuracy
    confusion = confusionmat(label_test, label_test_pred);
    figure(2);
    plotconfusion(categorical(label_test), categorical(label_test_pred),...
        ["BoW + KNN"]);
    accuracy = sum(label_test == label_test_pred) / n_test;
end
