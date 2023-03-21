function data = ddpi_offline(videoFile, options, dDPIParams)
    % This function applies offline digital Dual Purkinje Image (dDPI) algorithm 
    % to an input video and tracks eye.
    %
    % The dDPI algorithm track the Purkinje images by template matching and 
    % localize their position by a rapid and preise method that can be 
    % implemented on GPU. This localization method is developed by Parthasraty 
    % 2012. Please see https://doi.org/10.1038/nmeth.2071 for further information
    %
    % EXAMPLE USAGE:
    %  ddpi_offline(videoFileName)
    %
    %  ddpi_offline(videoFileName, Name, Value)
    %
    % INPUT:
    % videoFile string : the absolute path of the raw video file (avi)
    %
    % Name-Value Arguments:
    % - VideoFrameSize     (1,2) int: Size in pixels of video frame
    % - Visualize          boolean  : If true, show each frame post-processing
    % - DownSampling       int      : Downsampling factor for template matching
    % - GaussianKernelSize int      : Square kernel size used for blurring
    % - PauseDuration      double   : Pause time in seconds between each frame
    % - P1Threshold        int      : Intensity threshold of P1 mask
    % - P1ROI              int      : Size in pixels of square P1 ROI
    % - P4ROI              int      : Size in pixels of square P4 ROI
    % - P4Intensity        int      : Intensity of P4 template
    % - P4Radius           int      : Radius of P4 template
    % - P4TemplateSize     int      : Size in pixels of square P4 template
    %
    % OUTPUT:
    % data.trace        : x,y-coordinates of eye trace (P4-P1)
    % data.p1.trace     : x,y-coordinates of P1 center
    % data.p1.intensity : P1 intensity
    % data.p1.roi       : Coordinates of top left corner of P1 ROI
    % data.p4.trace     : x,y-coordinates of P4 center
    % data.p4.intensity : P4 intensity
    % data.p4.radius    : P4 radius
    % data.p4.score     : P4 template matching score
    % data.p4.roi       : Coordinates of top left corner of P4 ROI
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Validate arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        arguments
            videoFile string
            options.VideoFrameSize (1,2) double = [1080 1920]
            options.Visualize (1,1) logical = true;
            dDPIParams.DownSampling (1,1) double = 8
            dDPIParams.GaussianKernelSize (1,1) double = 3
            dDPIParams.PauseDuration (1,1) double = 0.01
            dDPIParams.P1Threshold (1,1) double = 250
            dDPIParams.P1ROI (1,1) double = 128
            dDPIParams.P4ROI (1,1) double = 64
            dDPIParams.P4Intensity (1,1) double = 150
            dDPIParams.P4Radius (1,1) double = 8
            dDPIParams.P4TemplateSize (1,1) double = 4
        end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Initial setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        %%% Read vidoe file
        v = rawVideoReader(...
            videoFile, 
            options.VideoFrameSize(2), 
            options.VideoFrameSize(1));
        
        %%% Get image format
        dsImgSize = options.VideoFrameSize / dDPIParams.DownSampling;
        nFrames   = round(v.totalFrames);
        
        %%% Setup visualization
        if options.Visualize
            vfig = figure();
            handles = imshow(zeros(options.VideoFrameSize), [0 255]); 
            hold on;
            handleP1ROI = plot(
                handles.Parent, 
                [0], [0], 
                '-', 
                'color', 'r', 
                'LineWidth', 2);
            handleP4ROI = plot(
                handles.Parent, 
                [0], [0], 
                '-', 
                'color', 'r', 
                'LineWidth', 2);
            hold off;
        end
        
        %%% Setup dDPI variables
        [xIdx, yIdx] = meshgrid(1:dsImgSize(2), 1:dsImgSize(1));
        [p1xIdx, p1yIdx] = meshgrid(1:dDPIParams.P1ROI, 1:dDPIParams.P1ROI);
        [p4xIdx, p4yIdx] = meshgrid(1:dDPIParams.P4ROI, 1:dDPIParams.P4ROI);
        
        p4Template = fspecial(
            'gaussian', 
            dDPIParams.P4TemplateSize, 
            dDPIParams.P4Radius / dDPIParams.DownSampling);
        p4Template = round(
            p4Template / max(p4Template, [], 'all') * dDPIParams.P4Intensity);
    
        tMatcher = vision.TemplateMatcher(
            'Metric', 'Sum of squared differences',
            'OutputValue', 'Metric matrix');
        
        %%% Initialize output data frame
        data.p1.trace = zeros(2, nFrames);
        data.p1.intensity = zeros(1, nFrames);
        data.p1.roi = zeros(2, nFrames);
        data.p4.trace = zeros(2, nFrames);
        data.p4.intensity = zeros(1, nFrames);
        data.p4.radius = zeros(1, nFrames);
        data.p4.score = zeros(1, nFrames);
        data.p4.roi = zeros(2, nFrames);
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Main tracking loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        for frameIdx=1:nFrames
    
            %%% Get next frame %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            frame = v.getNextFrame;
            gImg = frame(:, :, 1);
            dsImg = gImg(
                1:dDPIParams.DownSampling:end,
                1:dDPIParams.DownSampling:end,:);
            cdsImg = dsImg;
            
            %%% P1 estimation and localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Threshold to evaluate P1
            p1MaskImg = dsImg;
            p1Mask = p1MaskImg < dDPIParams.P1Threshold;
            p1MaskImg(p1Mask) = 0;
    
            % Estimate P1 ROI by center of mass
            p1c0 = double(sum(p1MaskImg, 'all'));
            p1cX = double(sum(p1MaskImg .* xIdx, 'all')) / p1c0;
            p1cY = double(sum(p1MaskImg .* yIdx, 'all')) / p1c0;
            p1Loc = round(dDPIParams.DownSampling * [p1cX, p1cY]);
    
            [p1ROIx, p1ROIxEnd, p1ROIy, p1ROIyEnd] = ...
                getROI(p1Loc, dDPIParams.P1ROI, options.VideoFrameSize);
    
            % Extract P1 ROI and apply Gaussian blur
            p1ROI = gImg(p1ROIy:p1ROIyEnd, p1ROIx:p1ROIxEnd);
            p1ROI = imgaussfilt(double(p1ROI), dDPIParams.GaussianKernelSize);
    
            % Localize P1 center by center of mass
            p10 = sum(p1ROI, 'all');
            p1x = sum(p1ROI .* p1xIdx, 'all') / p10;
            p1y = sum(p1ROI .* p1yIdx, 'all') / p10;
            
            % Output P1 information
            data.p1.trace(1, frameIdx) = p1x + double(p1ROIx);
            data.p1.trace(2, frameIdx)  = p1y + double(p1ROIy);
            data.p1.intensity(frameIdx) = ...
                p10 / dDPIParams.P1ROI / dDPIParams.P1ROI;
            data.p1.roi(:, frameIdx) = [p1ROIx p1ROIy];
            
            %%% P4 estimation and localization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mask P1 ROI (assumes P4 is outside of P1 ROI)
            dsP1ROI = round(
                double([p1ROIx, p1ROIy]) / dDPIParams.DownSampling * 0.875);
            dsP1ROIEnd = round(
                double([p1ROIxEnd, p1ROIyEnd]) / dDPIParams.DownSampling * 1.125);
            cdsImg(dsP1ROI(2):dsP1ROIEnd(2), dsP1ROI(1):dsP1ROIEnd(1)) = 0;
            
            % Estimate P4 ROI by template matching
            p4Error = tMatcher(cdsImg, p4Template);
            [p4Score, p4ErrorIdx]  = min(p4Error(:));
            [p4LocY, p4LocX] = ind2sub(size(p4Error), p4ErrorIdx);
            p4Loc = round(dDPIParams.DownSampling * [p4LocX, p4LocY]);
    
            [p4ROIx, p4ROIxEnd, p4ROIy, p4ROIyEnd] = ...
                getROI(p4Loc, dDPIParams.P4ROI, options.VideoFrameSize);
    
            % Extract P4 ROI and fine-tune P4 ROI estimation with COM
            p4ROI = double(gImg(p4ROIy:p4ROIyEnd, p4ROIx:p4ROIxEnd));
    
            p4ROI4 = power(p4ROI, 4);
            p40 = sum(p4ROI4, 'all');
            p4cx = round(sum(p4ROI4 .* p4xIdx, 'all') / p40);
            p4cy = round(sum(p4ROI4 .* p4yIdx, 'all') / p40);
            p4Loc = [p4cx + p4ROIx, p4cy + p4ROIy];
    
            [p4ROIx, p4ROIxEnd, p4ROIy, p4ROIyEnd] = ...
                getROI(p4Loc, dDPIParams.P4ROI, options.VideoFrameSize);
    
            % Reextract P4 ROI and apply Gaussian blur
            p4ROI = double(gImg(p4ROIy:p4ROIyEnd, p4ROIx:p4ROIxEnd));
            sp4ROI = imgaussfilt(p4ROI, dDPIParams.GaussianKernelSize);
    
            % Localize P4 center by radial symmetry
            [p4x, p4y, p4Radius] = radialcenter(sp4ROI);
            
            % Output P4 information
            data.p4.trace(1, frameIdx) = p4x + double(p4ROIx);
            data.p4.trace(2, frameIdx) = p4y + double(p4ROIy);
            data.p4.radius(frameIdx) = p4Radius;
            data.p4.score(frameIdx) = p4Score;
            data.p4.intensity(frameIdx) = p40 / dDPIParams.P4ROI / dDPIParams.P4ROI; 
            data.p4.roi(:, frameIdx) = [p4ROIx p4ROIy];
            
            %%% Visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if (options.Visualize)
                set(handles, 'CData', frame);
                set(handleP1ROI, ...
                    'XData', [p1ROIx, p1ROIxEnd, p1ROIxEnd, p1ROIx, p1ROIx], ...
                    'YData', [p1ROIy, p1ROIy, p1ROIyEnd, p1ROIyEnd, p1ROIy]);
                 set(handleP4ROI, ...
                    'XData', [p4ROIx, p4ROIxEnd, p4ROIxEnd, p4ROIx, p4ROIx], ...
                    'YData', [p4ROIy, p4ROIy, p4ROIyEnd, p4ROIyEnd, p4ROIy]);
                drawnow();
            end
            pause(dDPIParams.PauseDuration);
        end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% End of loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        %%% Output DPI trace
        data.trace = data.p4.trace - data.p1.trace;
        
        %%% Clean up
        if (options.Visualize)
            close(vfig);
        end
        fprintf("Tracking finished\n");
    end
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % getROI
    % Draws bounding box given center and ROI size within full image
    %
    % Inputs:
    % - loc     (1,2) array : center coordinates within full image
    % - boxSize (1,1) int   : size in pixels of square ROI
    % - imgSize (1,2) array : full image size
    % Outputs:
    % - ROIx    : min x-coordinate of ROI within full image
    % - ROIxEnd : max x-coordinate of ROI within full image
    % - ROIy    : min y-coordinate of ROI within full image
    % - ROIyEnd : max y-coordinate of ROI within full image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [ROIx, ROIxEnd, ROIy, ROIyEnd] = getROI(loc, boxSize, imgSize)
        ROIx = loc(1) - boxSize / 2 + 1;
        ROIx = max(1, ROIx);
        ROIxEnd = ROIx + boxSize - 1;
        ROIxEnd = min(imgSize(2), ROIxEnd);
        
        ROIy = loc(2) - boxSize / 2 + 1;
        ROIy = max(1, ROIy);
        ROIyEnd = ROIy + boxSize - 1;
        ROIyEnd = min(imgSize(1), ROIyEnd);
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % radialcenter
    % Finds intensity center of ROI using the radial symmetry algorithm
    %
    % Inputs:
    % - I (n,n) array : ROI of image
    % Outpus:
    % - xc    : center x-coordinate
    % - yc    : center y-coordinate
    % - sigma : "Particle width" or estimated radius
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [xc, yc, sigma] = radialcenter(I)
    
        % Number of grid points
        [Ny, Nx] = size(I);
        h = ones(3) / 9;  % Simple 3x3 averaging filter
        
        % Grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
        % Grid midpoint coordinates are -n+0.5:n-0.5;
        xm_onerow = -(Nx - 1) / 2.0 + 0.5:(Nx - 1) / 2.0 - 0.5;
        xm = xm_onerow(ones(Ny-1, 1), :);
        % Note that y increases "downward"
        ym_onecol = (-(Ny-1) / 2.0 + 0.5:(Ny-1) / 2.0 - 0.5)';  
        ym = ym_onecol(:,ones(Nx-1,1));
    
        % Calculate derivatives along 45-degree shifted coordinates (u and v)
        % Note that y increases "downward" (increasing row number) -- used
        % when calculating "m" below.
        dIdu = I(1:Ny-1, 2:Nx) - I(2:Ny, 1:Nx-1);
        dIdv = I(1:Ny-1, 1:Nx-1) - I(2:Ny, 2:Nx);
        
        % Smoothing 
        fdu = conv2(dIdu, h, 'same');
        fdv = conv2(dIdv, h, 'same');
        dImag2 = fdu.*fdu + fdv.*fdv; % gradient magnitude, squared
    
        % Slope of the gradient. Note that we need a 45 degree rotation of 
        % the u,v components to express the slope in the x-y coordinate system.
        % The negative sign "flips" the array to account for y increasing
        % "downward"
        m = -(fdv + fdu) ./ (fdu-fdv); 
    
        % *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
        % zero. In this case, replace with the un-smoothed gradient.
        NNanm = sum(isnan(m(:)));
        if NNanm > 0
            unsmoothm = (dIdv + dIdu) ./ (dIdu-dIdv);
            m(isnan(m))=unsmoothm(isnan(m));
        end
        % If it's still NaN, replace with zero. (Very unlikely.)
        NNanm = sum(isnan(m(:)));
        if NNanm > 0
            m(isnan(m))=0;
        end
    
        % Almost as rarely, an element of m can be infinite if the smoothed u and v
        % derivatives are identical. To avoid NaNs later, replace these with some
        % large number -- 10x the largest non-infinite slope. The sign of the
        % infinity does not matter
        try
            m(isinf(m))=10 * max(m(~isinf(m)));
        catch
            % If this fails, it's because all the elements are infinite. Replace
            % with the unsmoothed derivative.
            m = (dIdv + dIdu) ./ (dIdu-dIdv);
        end
    
        % Shorthand "b", which is the y intercept of the line of slope m that 
        % goes through each grid midpoint
        b = ym - m .* xm;
    
        % Weighting: weight by square of gradient magnitude and inverse 
        % distance to gradient intensity centroid.
        sdI2 = sum(dImag2(:));
        xcentroid = sum(dImag2.*xm, 'all') / sdI2;
        ycentroid = sum(dImag2.*ym, 'all') / sdI2;
        w  = dImag2./sqrt((xm-xcentroid).*(xm-xcentroid)+(ym-ycentroid).*(ym-ycentroid));  
    
        % Least-squares minimization to determine the translated coordinate
        % system origin (xc, yc) such that lines y = mx+b have
        % the minimal total distance^2 to the origin.
        % See function lsradialcenterfit (below)
        [xc, yc] = lsradialcenterfit(m, b, w);
    
        % Return output relative to upper left coordinate
        xc = xc + (Nx+1)/2.0;
        yc = yc + (Ny+1)/2.0;
    
        % A rough measure of the particle width.
        % Not connected to center determination, but may be useful for tracking applications 
        % Could eliminate for (very slightly) greater speed
        Isub = I - min(I(:));
        [px,py] = meshgrid(1:Nx,1:Ny);
        xoffset = px - xc;
        yoffset = py - yc;
        r2 = xoffset.*xoffset + yoffset.*yoffset;
        % Second moment is 2*Gaussian width
        sigma = sqrt(sum( sum(Isub .*r2)) / sum(Isub(:) )) / 2;  
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % lsradialcenterfit
    % Least-squares minimization to determine coordinate system
    % origin that minimize error from line with given parameters
    %
    % Inputs:
    % - m : Gradients in ROI
    % - b : y-intercepts of each gradient through ROI mid-point
    % - w : Weighting by square of gradient magnitude and inverse distance to gradient intensity centroid
    % Outputs:
    % - xc : Optimal origin x-coordinate
    % - yc : Optimal origin y-coordinate
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [xc, yc] = lsradialcenterfit(m, b, w)
        wm2p1 = w./(m.*m+1);
        sw  = sum(wm2p1, 'all');
        smmw = sum(m.*m.*wm2p1, 'all');
        smw  = sum(m.*wm2p1, 'all');
        smbw = sum(m.*b.*wm2p1, 'all');
        sbw  = sum(b.*wm2p1, 'all');
        det = smw*smw - smmw * sw;
        xc = (smbw*sw - smw*sbw) / det;   % relative to image center
        yc = (smbw*smw - smmw*sbw) / det; % relative to image center
    
    end
    
