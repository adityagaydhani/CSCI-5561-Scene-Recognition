function [feature] = GetTinyImage(I, output_size)
    % Input: I - An image
    %        output_size - Size of output image. Dim: w x h
    % 
    % Output: feature - Tiny features of image I. Dim: w*h x 1
    % 
    % Description: Generates tiny features from image I
    
    w = output_size(2); h = output_size(1);
    I_double = im2double(I);
    I_resized = imresize(I_double, [w h]);  % Dim: w x h
    I_vectorized = I_resized(:); % Dim: w*h x 1
    I_mean = mean(I_vectorized);
    I_zero_mean = I_vectorized - I_mean; % Dim: w*h x 1
    
    % Normalize to have 0 mean and unit length
    feature = I_zero_mean / norm(I_zero_mean); % Dim: w*h x 1
end
    