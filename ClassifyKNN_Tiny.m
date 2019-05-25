function [confusion, accuracy] = ClassifyKNN_Tiny
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
    % Get tiny features from training images
    file_name = './objects/tinyFeatures.mat';
    
    if isfile(file_name)
        load(file_name, 'features_train');
    else
        w = 16; h = 16;
        features_train = zeros(n_train, w*h);
        for i = 1 : n_train
            I = training_image_cell{i};
            features_train(i, :) = transpose(GetTinyImage(I, [w h]));
        end
    end
    
    % Get tiny features from testing images
    if isfile(file_name)
        load(file_name, 'features_test');
    else
        w = 16; h = 16;
        features_test = zeros(n_test, w*h);
        for i = 1 : n_test
            I = testing_image_cell{i};
            features_test(i, :) = transpose(GetTinyImage(I, [w h]));
        end
    end
    
    save(file_name, 'features_train', 'features_test');
    
    %%
    % Predict labels for test images
    label_test_pred = PredictKNN(features_train, label_train,...
        features_test, 10);
    
    % Compute confusion matrix and accuracy
    confusion = confusionmat(label_test, label_test_pred);
    figure(1);
    plotconfusion(categorical(label_test), categorical(label_test_pred),...
        ["Tiny + KNN"]);
    accuracy = sum(label_test == label_test_pred) / n_test;
end