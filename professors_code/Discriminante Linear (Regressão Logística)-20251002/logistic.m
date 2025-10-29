%Aprendizado de Máquina - ENE/UnB
%Prof. Daniel Guerreiro e Silva
%EXEMPLO - DISCRIMINANTE LOGISTICO

% clear;
% close all;

load iris.mat

%poe uma coluna de 1s nos dados, para associar ao coef. w0 do discriminante
X = [ ones(1,size(irisInputs,2));
      irisInputs];
R = irisTargets;

split = .7; %training/test split
N = length(X); %dataset length

sorteio = randperm(N); %embaralha a ordem das amostras

Xtr = X(:,sorteio(1:N*split)); %sorteia para conj. treino
Rtr = R(:,sorteio(1:N*split));
Ntrain = size(Xtr,2);

Xts = X(:,sorteio(N*split+1:end)); %restante vai para conj. teste
Rts = R(:,sorteio(N*split+1:end));
Nteste = size(Xts,2);

d = size(X,1); %dimensao da entrada (numero de atributos + bias)
K = size(R,1); %numero de classes (numero de funcoes discriminantes)

epochs = 500; %numero de iteracoes (epocas) do metodo de descida do gradiente
eta = 0.1; %taxa de aprendizado

W = rand(K,d)*0.02 - 0.01; %inicializacao dos coeficientes

errorRate = zeros(1,epochs);
Y = zeros(K,Ntrain);
Yteste = zeros(K,Nteste);

%TREINAMENTO
for ep=1:epochs

  dW = zeros(K,d);

  for t=1:Ntrain

    g = W*Xtr(:,t); %calcula os K discriminantes em forma matricial

    for k=1:K
      y(k) = exp(g(k))/sum(exp(g)); %saida classificador (softmax)
      error = Rtr(k,t)-y(k); %erro
      dW(k,:) = dW(k,:) + error*Xtr(:,t)'; %acumula passo de ajuste
    end
    Y(:,t) = y';%guarda a predicao do classificador
  end

  %ERRO TREINAMENTO
  errorRate(ep) = sum(sum(abs((Y>(1/K))-Rtr))~=0)/Ntrain;
  if(mod(ep,10)==0)
    printf("Epoca %d: Taxa de erro = %.1f%%\n", ep, errorRate(ep)*100);
  end

  %PASSO DE AJUSTE
  W = W + eta.*dW;
end

figure(1);plot(1:epochs,errorRate);xlabel('Iteracoes', 'FontSize', 14);ylabel('Taxa de erro - Treinamento', 'FontSize', 14);

%ANALISE NO CONJUNTO DE TESTE
Gteste = W*Xts;
for k=1:K
    Yteste(k,:) = exp(Gteste(k,:))./sum(exp(Gteste)); %saida classificador
end
errorRateTeste = sum(sum(abs((Yteste>(1/K))-Rts))~=0)/Nteste;

fprintf(1,'ERRO TREINAMENTO: %.1f%%, ERRO TESTE: %.1f%%\n', errorRate(end)*100, errorRateTeste*100);
  