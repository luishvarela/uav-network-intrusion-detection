# Deep Learning para detecção e robustez de ataques em redes de VANTs

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![USP](https://img.shields.io/badge/ICMC-USP-0055A5?style=for-the-badge)](https://www.icmc.usp.br/)

Este repositório reúne duas etapas de um trabalho sobre classificação de tráfego de rede associado a ataques contra **Veículos Aéreos Não Tripulados (VANTs / drones)**:

1. Treinamento e avaliação de modelos de Deep Learning para classificar tráfego normal e identificar classes de ataque.
2. Avaliação da robustez desses modelos diante de perturbações adversariais.

## Projetos

### [Classificação de tráfego com redes neurais](deep-learning-redes/README.md)

Compara MLP, RNN, CNN, GRU e LSTM nas tarefas binária e multiclasse. Inclui preparação dos dados, tuning de hiperparâmetros, métricas, matrizes de confusão e instruções de reprodução.

### [Avaliação de ataques adversariais](ataques-adversariais/README.md)

Aplica FGSM, PGD e ruído Gaussiano aos modelos, comparando o desempenho antes e depois das perturbações. Inclui resultados por intensidade, gráficos e limitações da avaliação.

## Relação entre as etapas

A avaliação adversarial reutiliza o dataset, os scalers, os encoders e os modelos gerados pelo projeto de classificação. Para reproduzir o fluxo completo, comece pelo [`deep-learning-redes/`](deep-learning-redes/README.md): execute o tuning se necessário e depois os notebooks de treinamento. Com os modelos atuais disponíveis, execute [`ataques_adversariais.ipynb`](ataques-adversariais/ataques_adversariais.ipynb).

Cada etapa mantém seus resultados atuais em pastas estáveis. Uma nova execução atualiza esses arquivos, sem criar cópias datadas de execuções anteriores. A classificação de referência é mantida separada da versão com dados tratados.

## Dataset e reprodução

O dataset de referência está disponível no [Figshare](https://figshare.com/articles/dataset/Unmanned_Aerial_Vehicles-Network_Intrusion_Detection_Dataset/25486462). Os detalhes de uso, dependências, comandos e limitações estão nos READMEs de cada projeto.

**Citação do dataset:** Hadi, H. J. (2025). *Unmanned Aerial Vehicles-Network Intrusion Detection Dataset* (v2) [Dataset]. Figshare. https://doi.org/10.6084/m9.figshare.25486462.v2.

O dataset está disponível sob a licença [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). O arquivo `Normal+Attacks.csv` é proveniente desse dataset. Ao reutilizar ou redistribuir o conjunto, mantenha a atribuição acima e indique alterações feitas. O código e os notebooks desenvolvidos para este projeto estão sob a licença [MIT](LICENSE); essa licença não se aplica ao dataset.

Os resultados e as ressalvas sobre a comparação com o artigo associado ao dataset estão em [deep-learning-redes/README.md](deep-learning-redes/README.md#comparação-com-o-artigo-de-referência).

## Autoria

Desenvolvido por **Luís Henrique Varela Medeiros Bezerra**  
Bacharelado em Ciência da Computação (BCC) — **ICMC / USP**  
GitHub: [@luishvarela](https://github.com/luishvarela)
