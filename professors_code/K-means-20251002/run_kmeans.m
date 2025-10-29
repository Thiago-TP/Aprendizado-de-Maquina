clear all; clc; close all;

load iris.mat

X = irisInputs(2:3,:)';
k = 2; % número de clusters

[Means, Assignments, Error] = kmeans(X, k);

disp('Centróides:');
disp(Means);
disp('Erro de reconstrução:');
disp(Error);