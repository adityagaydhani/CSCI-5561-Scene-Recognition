function [image_cell] = getImageCell(image_paths)
    % Input: image_paths - A string array consisting of image paths
    %
    % Output: image_cell - A array consisting of image cells for every
    %                      images specified in image_paths.
    %
    % Description: This function reads images from each path present in
    %              image_paths and stores them in a array of image cells.
    
    n = size(image_paths, 1);
    image_cell = cell(n, 1);
    for i = 1 : n
        I = imread(image_paths(i));
        image_cell{i} = I;
    end
end