function [bow_feature] = ComputeBoW(feature, vocab)
    % Input: feature - Dense SIFT features for an image. Dim: p x 128
    %        vocab - Dictionary for BoW, consisting of kmeans centroids.
    %                Dim: 50 x 128
    %
    % Output: bow_feature - Normalized BoW histogram. Dim: 50 x 1
    %
    % Description: This function converts dense SIFT features for an image
    %              to a BoW histogram. It finds the nearest center in vocab
    %              for each SIFT feature and keeps track of the count of
    %              dense SIFT features belonging to each center of vocab.
    
    % Get knn indexes of vocab for every dense SIFT feature. Dim: p x 1
    knn = knnsearch(double(vocab), double(feature), 'K', 1);
    
    % For every knn index found, increment the count of the corresponding
    % centroid index to generate histogram.
    bow_feature = zeros(size(vocab, 1), 1); % 50 x 1
    for i = 1 : size(knn, 1)
        bow_feature(knn(i)) = bow_feature(knn(i)) + 1;
    end
    
    % Normalize histogram to have unit length
    bow_feature = bow_feature / norm(bow_feature);
end