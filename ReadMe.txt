1 - O input são sons no formato .wav

2 - No passo seguinte, o ficheiro de som é dividido em timestamps de 5 segundos.

3 - Seguidamente é feita extração de features e o resultado é guardado num ficheiro csv.

4 - Por fim, utiliza-se o modelo XGBoost e é imprimida uma tabela com as primeiras 3 previsões para cada um dos slices do som e é imprimida uma média das primeiras previsões, dando-se assim uma previsão final para o som inteiro.
