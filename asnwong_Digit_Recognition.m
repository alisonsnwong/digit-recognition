% Name: Alison Wong
% SID: 918258892

% Step 1: Loading and Understanding Data
% a. load data
load USPS;

% b. display the first 16 images in train patterns
figure;
for k = 1:16
    subplot(4, 4, k);
    imagesc(reshape(train_patterns(:, k), 16, 16)');
    colormap(gray);
    axis off;
end

% Save the figure
saveas(gcf, 'first_16_images.png');

% Step 2: Compute the mean digits in the train patterns
% Initialize train_aves matrix
train_aves = zeros(256, 10);

% Compute the mean digits for each class
for k = 1:10
    train_aves(:, k) = mean(train_patterns(:, train_labels(k, :)==1), 2);
end

% Display the mean digit images
figure;
for k = 1:10
    subplot(2, 5, k);
    imagesc(reshape(train_aves(:, k), 16, 16)');
    colormap(gray);
    axis off;
end

% Save the figure as a PDF file
saveas(gcf, 'mean_digit_images.pdf');

% Step 3: Simplest classification computation
% a. initialize test classif matrix
test_classif = zeros(10, 4649);

% Compute the squared Euclidean distances between test patterns and mean digit images
for k = 1:10
    test_classif(k, :) = sum((test_patterns - repmat(train_aves(:, k), [1, 4649])).^2);
end

% b. compute classification results
[~, test_classif_res] = min(test_classif);

% c. initialize confusion matrix
test_confusion = zeros(10);

% Compute confusion matrix
for k = 1:10
    tmp = test_classif_res(test_labels(k, :) == 1);
    for j = 1:10
        test_confusion(k, j) = sum(tmp == j);
    end
end

% Display confusion matrix
disp('Confusion Matrix:');
disp(test_confusion);

accuracy = sum(diag(test_confusion)) / sum(test_confusion(:));

disp('Accuracy:');
disp(accuracy);

% Step 4: SVD-based classification
% Initialize train u array
train_u = zeros(256, 17, 10);

% a. compute rank 17 SVD for each digit
for k = 1:10
    [train_u(:,:,k), ~, ~] = svds(train_patterns(:, train_labels(k,:) == 1), 17);
end

% b. compute expansion coefficients for each test digit image
test_svd17 = zeros(17, 4649, 10);
for k = 1:10
    test_svd17(:,:,k) = train_u(:,:,k)' * test_patterns;
end

% c. 
% compute approximation errors between each original test digit image and 
% its rank 17 approximation using the kth digit images in the training data set
test_svd17res = zeros(10, 4649);
for k = 1:10
    approximations = train_u(:,:,k) * test_svd17(:,:,k);
    errors = sum((test_patterns - approximations).^2);
    test_svd17res(k, :) = errors;
end

% d. initialize confusion matrix
test_svd17_confusion = zeros(10);

% Compute confusion matrix
for k = 1:10
    [~, test_classif_res] = min(test_svd17res);
    tmp = test_classif_res(test_labels(k,:) == 1);
    for j = 1:10
        test_svd17_confusion(k, j) = sum(tmp == j);
    end
end

% Display confusion matrix
disp('SVD-based Confusion Matrix:');
disp(test_svd17_confusion);