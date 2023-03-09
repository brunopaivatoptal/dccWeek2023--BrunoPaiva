**Autor : Bruno Barbosa Miranda de Paiva**
Para executar o pipeline completo, use o arquivo *generateFinalSubmission.py*.

Este código permite treino usando GPU em ambas as fases. 

De forma resumida, o processo ocorre dessa forma:
![[Pasted image 20230309080443.png]]

A partir do sinal bruto, fazemos data augmentation a cada batch, passamos esse dado para o pré-treino de uma rede neural (treinada na tarefa de classificação), e, por fim, para a camada de embedding (penúltima camada da rede), que então serve de entrada para o treinamento de um algoritmo de boosting. 

A arquitetura final da rede neural foi algo assim:
![[Pasted image 20230309080646.png]]

O módulo de embedding de sinal é desenhado para ser capaz de quebrar um sinal em "wavelets" e transferir isso para um vocabulário de tamanho fixo:
![[Pasted image 20230309080734.png]]

Esse vocabulário é o ponto de entrada para a arquitetura da rede neural deste trabalho.
