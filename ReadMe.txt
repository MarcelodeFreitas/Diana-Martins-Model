# Diana Martins Model Preparation for FMdeploy

This GitHub repository contains the code for adapting the "Deep Learning for Real-time Decoding of Sound Events Related to Autism Spectrum Disorder" model developed by D. Martins, as presented in the dissertation submitted to the University of Minho in 2021.

The purpose of this adaptation is to integrate the original model with FMdeploy, a machine learning model deployment framework. By adapting the model code and configuration, it can be seamlessly deployed and utilized within the FMdeploy ecosystem.

## References
- D. Martins, "Deep Learning for Real-time Decoding of Sound Events Related to Autism Spectrum Disorder," University of Minho, 2021.

Please refer to the original dissertation for more details on the model and its implementation.




1 - O input são sons no formato .wav

2 - No passo seguinte, o ficheiro de som é dividido em timestamps de 5 segundos.

3 - Seguidamente é feita extração de features e o resultado é guardado num ficheiro csv.

4 - Por fim, utiliza-se o modelo XGBoost e é imprimida uma tabela com as primeiras 3 previsões para cada um dos slices do som e é imprimida uma média das primeiras previsões, dando-se assim uma previsão final para o som inteiro.
