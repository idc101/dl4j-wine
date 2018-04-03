function X = plotData(D, cols)

% down sample
% nrows = size(D,1);
% nrand = 200; % Choose 1000 rows
% rand_rows = randperm(nrows, nrand);
% D = D(rand_rows,:);

% Create New Figure
figure;

y = D(:, 1);
X = D(:, 2:end);

% for n = 2:13
%     subplot(4, 4, n);
%     scatter(y, D(:, n));
%     %ylabel(cols{n});
% end

% % Plot Examples
one = find(y == 1);
two = find(y == 2);
three = find(y == 3);
hold on;
plot(X(one, 8), X(one, 11), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(two, 8), X(two, 11), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
%plot(X(three, 8), X(three, 11), 'k-', 'LineWidth', 2, 'MarkerSize', 7);
hold off;

% xlabel ('% change');
% ylabel ('quality');

end
