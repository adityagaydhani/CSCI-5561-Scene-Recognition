function [vocab] = BuildVisualDictionary(training_image_cell, dic_size)
    % Input: training_image_cell - An array consisting of image cells of
    %                              training images. Dim: n_train x 1
    %        dic_size - Size of dictionary for the vocabulary, and value of
    %                   k for kmeans algorithm.
    %
    % Output: vocab - Vocabulary generated from kmeans clustering.
    %                 Dim: dic_size x 128
    %
    % Description: This function extracts dense SIFT descriptors from all
    %              the images present in training_image_cell and builds a
    %              pool of dense SIFT descriptors. It then performs kmeans
    %              clustering to get k centroids from the pool and returns
    %              these centroids as vocabulary.
    
    pool = [];
    n_train = size(training_image_cell, 1);
    
    for i = 1 : n_train
        I = training_image_cell{i};
        I = single(vl_imdown(I));
        [~, d] = vl_dsift(I, 'size', 8, 'step', 2, 'fast'); % Dim: 128 x p
        pool = [pool; transpose(d)];
    end
    [~, C] = kmeans(double(pool), dic_size); % Dim: dic_size x 128
    vocab = C; % Dim: dic_size x 128
end