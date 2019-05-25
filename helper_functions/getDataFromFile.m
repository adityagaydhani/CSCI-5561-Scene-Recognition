function [label_enum, paths] = getDataFromFile(file_name)
    % Input: file_name - Name of the file from which data needs to be read
    % 
    % Output: label_enum - An array containing enumerated class label names
    %                      for all the correspoding samples from file_name.
    %                      Dim: n x 1
    %         paths - A string array containing the image paths from
    %                 file_name. Dim: n x 1
    %                 Note: The path format from file_name is modified
    %                 according to Linux specification and may not work on
    %                 Windows.
    % Description: This function reads lines from the file scpecified by
    %              file_name and returns enumerated labels and paths for
    %              all lines present in file.
    
    label_enum = []; paths = [];
    
    label_names = ["Office", "Kitchen", "LivingRoom", "Bedroom",...
        "Store", "Industrial", "TallBuilding", "InsideCity", "Street",...
        "Highway", "Coast", "OpenCountry", "Mountain", "Forest", "Suburb"];
    
    file_ID = fopen(file_name, 'r');
    
    tline = fgetl(file_ID);
    while ischar(tline)
        splits = strsplit(tline, ' ');
        label_name = splits(1); label_name = label_name{1};
        label_enum = [label_enum; find(label_names==label_name)];
        path = splits(2); path = path{1};
        path = strrep(path, '\', '/'); path = "./" + path;
        paths = [paths; path];
        tline = fgetl(file_ID);
    end
    
    fclose(file_ID);
end
