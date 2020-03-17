
%% read dataset

rootFolder = fullfile(pwd,'image_set');
imgSets = imageSet(rootFolder,'recursive');
{imgSets.Description} % display all labels on one line
[imgSets.Count] % show the corresponding count of imag


%% feature extraction (developed from tutorial of bag of features)
minSetCount = min([imgSets.Count]); % determine the smallest amount of images in a category
% Use partition method to trim the set.
imgSets = partition(imgSets, minSetCount, 'randomize');
% Notice that each set now has exactly the same number of images.
[imgSets.Count]

[trainingSets, validationSets] = partition(imgSets, 0.75, 'randomize');


bag = bagOfFeatures(trainingSets,'VocabularySize',500,'PointSelection','Detector');


%% Visualize Feature Vectors

figure
img = read(imgSets(1), 1);
featureVector = encode(bag, img);
subplot(3,2,1); imshow(img);
subplot(3,2,2);
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(imgSets(2), 1);
featureVector = encode(bag, img);
subplot(3,2,3); imshow(img);
subplot(3,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(imgSets(3), 1);
featureVector = encode(bag, img);
subplot(3,2,5); imshow(img);
subplot(3,2,6); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

figure
img = read(imgSets(4), 1);
featureVector = encode(bag, img);
subplot(3,2,1); imshow(img);
subplot(3,2,2); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(imgSets(5), 1);
featureVector = encode(bag, img);
subplot(3,2,3); imshow(img);
subplot(3,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(imgSets(6), 1);
featureVector = encode(bag, img);
subplot(3,2,5); imshow(img);
subplot(3,2,6); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

%% Create a Table using the encoded features


%Prepare data for classifierlearner
training_label = string.empty;
for i = 1:6
    for j = 1:trainingSets(1,i).Count
        training_label(end+1) = string(trainingSets(1,i).Description);
    end
end
objectdata = double(encode(bag, trainingSets));
ObjectImageData = array2table(objectdata);
ObjectImageData.Type = training_label';

%
[trainedModel, validationAccuracy] = trainClassifier(ObjectImageData);
disp("Validation accuracy of this trained classifier is"+ validationAccuracy);
%% Test out accuracy on test set! (developed from scene identification project)


objectdata = double(encode(bag, validationSets));
ObjectImageData = array2table(objectdata);
validation_label = string.empty;
for i = 1:6
    for j = 1:validationSets(1,i).Count
        validation_label(end+1) = string(validationSets(1,i).Description);
    end
end
actualType = validation_label';

predictedOutcome = trainedModel.predictFcn(ObjectImageData);
correctPredictions = 0;
for i = 1:length(predictedOutcome)
    correctPredictions = correctPredictions + strcmp(predictedOutcome(i),actualType(i));
end
validationAccuracy = correctPredictions/length(predictedOutcome) %#ok


%% Compared with the built-in classifer for image recognition (developed from tutorial of bag of features)
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

confMatrix = evaluate(categoryClassifier, validationSets);

img = imread(fullfile(rootFolder, 'camera', 'image_0048.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);
% Display the string label
categoryClassifier.Labels(labelIdx)