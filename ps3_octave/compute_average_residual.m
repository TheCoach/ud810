function residuals_mean = compute_average_residual(pts2d_projected, pts2d_expected, doPlot)
    residuals = sqrt(sumsq(pts2d_projected .- pts2d_expected, 2));
    residuals_mean = mean(residuals);
    if doPlot
        f = figure();
        subplot(2, 1, 1);
        scatter(pts2d_projected(:, 1), pts2d_projected(:, 2), 'r');
        scatter(pts2d_expected(:, 1), pts2d_expected(:, 2), 'b');
        subplot(2, 1, 2);
        plot(residuals);
        hold on;
        plot([0, length(residuals - 1)], [residuals_mean, residuals_mean], 'color', 'g');
        text(length(residuals) / 2, residuals_mean, num2str(residuals_mean));
        title('residuals (distances)');
        hold off;
    end
end
