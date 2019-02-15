function hough_lines_draw(img, outfile, peaks, rho, theta)
    % Draw lines found in an image using Hough transform.
    %
    % img: Image on top of which to draw lines
    % outfile: Output image filename to save plot as
    % peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    % rho: Vector of rho values, in pixels
    % theta: Vector of theta values, in degrees

    % TODO: Your code here
    f = figure();
    imshow(img);
    hold on;
    for p = 1 : length(peaks)
    	r = rho(peaks(p, 1));
    	t = theta(peaks(p, 2));
    	if sind(t) ~= 0 
    		x = linspace(0, size(img, 2));
    		y = (r - x * cosd(t)) / sind(t);
    	else 
    		y = linspace(0, size(img, 1));
    		x = (r - y * sind(t)) / cosd(t);
    	end
    	line(x, y, 'color', 'g');
    end
    hold off;
    saveas(f, outfile);
end
