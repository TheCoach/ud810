function f = drawEpiLines(epiLines, image)

    % four corners of the images
    p_ul = [0, 0, 1];
    p_bl = [0, size(image, 1), 1];
    p_ur = [size(image, 2), 0, 1];
    p_br = [size(image, 2), size(image, 1), 1];

    % left and right edge of the image
    ll = cross(p_ul, p_bl);
    lr = cross(p_ur, p_br);

    f = figure;
    imshow(image);
    hold on; 
    % intersection points of the edges and the epipolar line, which can be drawn ion the image
    for i = 1 : length(epiLines)
        p_l = cross(epiLines(:, i)', ll);
        p_r = cross(epiLines(:, i)', lr);
        p_l = p_l ./ p_l(3);
        p_r = p_r ./ p_r(3);
        line([p_l(1), p_r(1)], [p_l(2), p_r(2)], 'color', 'b');
    end
    hold off;
end