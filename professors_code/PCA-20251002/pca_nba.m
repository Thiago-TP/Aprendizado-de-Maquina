%Aprendizado de Máquina - ENE/UnB
%Prof. Daniel Guerreiro e Silva
%Aplicação de PCA no ranking de jogadores da NBA

close all;
clear all;

%reading CSV data
x = dlmread('nba.csv',';',1,0); %OCTAVE

printf('DATA:\n');
disp(x(1:10,:));
pause;
%selecting data columns
X = x(:, 3:end);
ids = x(:,1);

%data normalization
sigma = std(X);
m = mean(X);
printf('MEAN:\n');
disp(m);
printf('STD:\n');
disp(sigma);
pause;

X0 = (X-m)./sigma;
printf('NORMALIZED DATA:\n');
disp(X0(1:10,:));
pause;

%covariance matrix of normalized data is calculated
S = cov(X0);
printf('COVARIANCE MATRIX:\n');
disp(S);
pause;

[W, D, ~] = eig(S); %eigenvectors and eigenvalues
printf('EIGENVECTORS:\n');
W *= -1;
disp(W);
printf('EIGENVALUES:\n');
disp(D);
pause;

w = (W(:,end));
z = X0*w; %1st principal component is calculated, for each player

[zsorted, podium] = sort(abs(z), 'descend'); %sorting according to the ABSOLUTE score
printf('TOP 5 PLAYERS:\n');
disp([ids(podium) zsorted]);
printf('WEIGHT OF EACH METRIC:\n');
pesos = w./sum(w);
disp(pesos);
