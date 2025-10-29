%Aprendizado de Máquina - ENE/UnB
%Prof. Daniel Guerreiro e Silva
%Função de Clusterização pelo algoritmo k-means
%Entrada:
%X: matriz Nxd de dados
%k: numero de clusters
%Saida:
%M: matriz k x d de centróides (médias) dos clusters
%B: matriz N x k de rótulos dos dados com os clusters
%error: erro de reconstrução dos dados a partir dos centróides
function [M, B, error] = kmeans(X,k)

[N,d] = size(X);
draw = randperm(N);
M = X(draw(1:k),:); %sorteia k pontos de X para inicializar os centróides
Mold = zeros(k,d);
stop = 0;
iter = 0;

if(d==2)
  figure(iter+1);scatter(X(:,1),X(:,2));hold on;plot(M(:,1),M(:,2),'rx');title(sprintf('Iteracao %d', iter));
end

while (~stop)

    %EXPECTATION Step
    B = zeros(N,k);
    D = zeros(N,k); %inicializa matriz de distancias de cada ponto para cada centroide
    for idk=1:k
        D(:,idk) = sqrt(sum((X - M(idk,:)).^2,2)); %calcula dist. euclidiana
    end
    [~, mindist] = min(D,[],2);
    for idk=1:k
      B(mindist==idk,idk) = 1; %marca com 1 qual é o cluster associado a cada ponto
    end

    %MAXIMIZATION Step
    Mold = M; %guarda centroides anteriores antes de atualizá-los
    for idk=1:k
        b = B(:,idk)==1; %pega os exemplos vinculados ao cluster idk
        M(idk,:) = sum(X(b,:),1)/sum(b);
    end

    iter = iter + 1;

    if(d==2)
        figure(iter+1);scatter(X(:,1),X(:,2));hold on;plot(M(:,1),M(:,2),'rx');title(sprintf('Iteracao %d', iter));
    end

    if(iter>1)
        delta = sqrt(sum((M-Mold).^2,2)); %distancia entre os valores atualizados e anteriores dos centroides
        if(abs(max(delta))<1e-6) %se maior mudanca foi menor que 10^-3, hora de encerrar
            stop = 1;
    end
end

%erro final de reconstrucao
error = 0;
for id=1:N
    error = error + sqrt(sum((X(id,:)-M(B(id,:)==1,:)).^2));
end
error = error/N;

end
