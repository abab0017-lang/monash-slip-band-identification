%% Separate Grain Shear Map Data - Visualization
% Make sure this file and all .npy data files are in the same folder:
% C:\Users\Asus\Desktop\Individual Shear Map
% Requires: readNPY.m in the same folder or on MATLAB path

clc; clear; close all;

% Step 1: Set folder path
data_folder = 'C:\Users\Asus\Desktop\Individual Shear Map';
cd(data_folder);  % Change MATLAB's working directory to this folder

% Step 2: Load one shear map (for example, grain 303)
file_name = 'grain 303 shear map.npy';  % <--- you can change this to any other grain file

% Check that the file exists
if ~isfile(fullfile(data_folder, file_name))
    error('File not found: %s', file_name);
end

% Step 3: Read the NPY file
E = readNPY(fullfile(data_folder, file_name));
E = squeeze(E);

% Check basic info
whos E
min_val = min(E(:));
max_val = max(E(:));
mean_val = mean(E(:), 'omitnan');
disp([min_val, max_val, mean_val]);

% Replace NaN and Inf with 0
E(~isfinite(E)) = 0;

% Optional: if values are tiny, scale them up
if max(E(:)) < 1e-6
    E = E * 1e6;
end

% Normalize
lo = prctile(E(:), 1);
hi = prctile(E(:), 99);
I  = (E - lo) ./ (hi - lo + eps);
I  = max(0, min(1, I));

figure('Color','w');
imagesc(I); axis image off; colormap parula; colorbar;
title(sprintf('Normalized Map: %s', file_name), 'Interpreter','none');
