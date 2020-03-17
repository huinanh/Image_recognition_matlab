function categoryClassifier = train_category()

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

%% Compared with the built-in classifer for image recognition (developed from tutorial of bag of features)
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

