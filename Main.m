clear; close all; clc;

addpath('./helper_functions');
[confusion_tiny, accuracy_tiny] = ClassifyKNN_Tiny;
fprintf("Tiny image accuracy: %f\n", accuracy_tiny);

[confusion_knn_bow, accuracy_knn_bow] = ClassifyKNN_BoW;
fprintf("KNN-BoW image accuracy: %f\n", accuracy_knn_bow);

[confusion_svm_bow, accuracy_svm_bow] = ClassifySVM_BoW;
fprintf("SVM-BoW image accuracy: %f\n", accuracy_svm_bow);
