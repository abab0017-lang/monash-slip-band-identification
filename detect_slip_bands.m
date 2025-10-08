%% detect_slip_bands.m
% Detect slip bands in strain map images
% Needs: Image Processing Toolbox + readNPY helpers
% Included in the folder: 

%Downloaded and placed in the same folder: 
% Image Processing Toolbox
% constructNPYheader.m
% detect_slip_bands.m
% readNPY.m
% readNPYheader.m
% strain map.npy

% 1. Develop a pixel-wise classification model. 
% Based on the identification result that we already have, 
% we are going to develop a classification model that tells us if a pixel 
% in the strain map belongs to a slip band or the matrix. In this case, the 
% existing algorithm can be largely accelerated since the radon transform of 
% the matrix pixels can be skipped.


%% Parameters to change 
sigma_blur   = 1.0;        % how much to blur the image (reduce noise)
                            % using the sigma blur to reduce the noise
line_len_px  = 21;         % length of line used to highlight bands
                            % This can be changed to suit 
sens_thresh  = 0.50;       % how sensitive the threshold is and will be
                            % the greater the sensitivity, the longer 
                            %it will take to load but can be made more
                            %precise
min_area_px  = 60;         % smallest object size to keep
                             % the greater the smallest object size, the longer 
                            %it will take to load but can be made more
                            %precise
min_branch   = 15;         % shortest skeleton branch to keep
fill_gap_px  = 10;         % connect nearby line pieces
min_len_px   = 25;         % shortest line to accept

%% Step 1: Loading the data and normalising it
fname = "strain map.npy";                 % file name
E = readNPY(fname); E = squeeze(E); %removes the  unnecessary singleton dimensions.
%readNPY already downloaded into the same folder

% scale values between 0 and 1 for viewing
lo = prctile(E(:), 1); % take the 1st percentile of all values in E (ignore extreme low outliers)

hi = prctile(E(:), 99); % take the 99th percentile of all values in E (ignore extreme high outliers)

I  = (E - lo) ./ (hi - lo + eps); % rescale the matrix so values lie between 0 and 1 (normalisation)

I  = max(0, min(1, I)); % clip any values below 0 to 0 and above 1 to 1


figure('Name','1. Input (normalised)'); % show the image in a figure window with a colour scale after downloading the image processing toolbox

imagesc(I); axis image off; colormap parula; colorbar;
% display the image
% equal aspect ratio, no axes
% apply the "parula" colour map
% Using the parula because: 
%It smoothly goes from dark blue → light blue → green → yellow.

%It avoids very dark or very bright extremes that can hide details.

% It's perceptually uniform, meaning equal changes in data look like equal changes in colour to the human eye.
% add a colourbar scale on the side

%% 2) Pre-Process before detection of slip bands after normalisation of data
% blur to reduce noise and increase contrast
I_s = imgaussfilt(I, sigma_blur); % applying a Gaussian blur to reduce noise in the image

% applying adaptive histogram equalisation (CLAHE) to improve local contrast
% 'ClipLimit' prevents over-amplifying noise
% 'rayleigh' distribution gives smoother enhancement
I_enh = adapthisteq(I_s, 'ClipLimit', 0.01, 'Distribution', 'rayleigh');

%Since, I want to the steps in the detection, two panels are created
figure('Name','2. Preprocess'); 
subplot(1,2,1); imshow(I_s,[]); title('Blurred'); % left panel: show the blurred image
subplot(1,2,2); imshow(I_enh,[]); title('Enhanced'); % right panel: show the contrast-enhanced image


%% Mid-Process: Line Enhancement 
% highlight thin, bright, line-like features at different angles
thetas = 0:15:165;     % define angles (0 to 165 degrees, in steps of 15) to scan for lines
                    
resp = zeros(size(I_enh), 'like', I_enh); % make an empty image to store the strongest line response

for th = thetas     % create a line-shaped structuring element at this angle

    se = strel('line', line_len_px, th);
    r  = imtophat(I_enh, se);           % use top-hat filter to highlight thin bright lines at this angle
       
    resp = max(resp, r);       % keep the maximum response across all angles (best line per pixel)
                 
end
resp = mat2gray(resp); % rescale result to [0,1] for viewing


figure('Name','3. Line-enhanced response'); imshow(resp,[]);% display the combined line-enhanced image


%% Identifying the Threshold and Refinement 
% make binary image and clean small noise
% turn the line-enhanced image into black & white using adaptive threshold
% 'ForegroundPolarity','bright' means we are looking for bright features
% 'Sensitivity' controls how strict the threshold is
BW = imbinarize(resp, 'adaptive', ...
    'ForegroundPolarity','bright', 'Sensitivity', sens_thresh);
% close small gaps in the detected bands using line-shaped elements
% this connects broken pieces along horizontal, vertical and diagonal directions
BW = bwareaopen(BW, min_area_px);
BW = imclose(BW, strel('line', 5, 0));% horizontal closing
BW = imclose(BW, strel('line', 5, 90));% vertical closing
BW = imclose(BW, strel('line', 5, 45));% diagonal (45°)
BW = imclose(BW, strel('line', 5, 135)); % diagonal (135°)

figure('Name','4. Binary after cleanup'); imshow(BW); 
% show the cleaned-up binary image

%% Step 5) Displaying the Lines
% thin the shapes down to their center lines

% reduce the binary shapes to thin center lines (skeleton)
% 'MinBranchLength' removes very short branches
BWsk = bwskel(BW, 'MinBranchLength', min_branch);

% remove small extra spurs or dangling ends from the skeleton

BWsk = bwmorph(BWsk, 'spur', 2);           
% show the skeleton image

figure('Name','5. Skeleton'); imshow(BWsk);

%% Hough Lines
% find straight line segments from skeleton

% apply the Hough transform to the skeleton image
% this finds evidence of straight lines at different angles and distances
[H,theta,rho] = hough(BWsk);

% pick the strongest peaks from the Hough transform (potential lines)
% here we select up to 30 peaks, and ignore weak ones below 30% of max
P = houghpeaks(H, 30, 'Threshold', ceil(0.3 * max(H(:))));


% convert those peaks back into actual line segments on the image
% 'FillGap' connects nearby pieces into one line
% 'MinLength' ignores very short lines
lines = houghlines(BWsk, theta, rho, P, ...
                   'FillGap',  fill_gap_px, ...
                   'MinLength', min_len_px);

%% FINAL PRINTING AND VISUALISATION
% show detected lines and save results

% show the original image

figure('Name','6. Detected Slip Bands'); imshow(I,[]); hold on;

% prepare a matrix to store the line coordinates (x1,y1,x2,y2)
coords = zeros(numel(lines),4);

% loop through all detected lines

for k = 1:numel(lines)
    p1 = lines(k).point1; p2 = lines(k).point2;
    plot([p1(1) p2(1)], [p1(2) p2(2)], 'r-', 'LineWidth', 2);
    plot(p1(1), p1(2), 'go', 'MarkerSize', 5, 'LineWidth',1);
    plot(p2(1), p2(2), 'go', 'MarkerSize', 5, 'LineWidth',1);
    coords(k,:) = [p1(1) p1(2) p2(1) p2(2)];
end
title(sprintf('Slip bands found: %d', size(coords,1)));

% add a title showing how many lines were found
% save the results:

writematrix(coords, 'slip_band_coords.csv');
imwrite(BWsk, 'slip_band_skeleton.png');
imwrite(resp, 'line_response.png');
disp('Saved: slip_band_coords.csv, slip_band_skeleton.png, line_response.png');
