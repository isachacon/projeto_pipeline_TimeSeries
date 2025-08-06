**Projeto de Previsão de Consumo de Energia ⚡**
----------------------------------------------
Este projeto implementa uma pipeline de Machine Learning para previsão de consumo de energia, utilizando dados históricos de consumo, clientes e clima. A pipeline é modularizada em scripts Python e utiliza DuckDB para processamento de dados e MLflow para rastreamento de experimentos e gerenciamento de modelos.

**Estrutura de Diretórios 📁**

A estrutura do projeto é organizada para clareza e manutenibilidade, seguindo as melhores práticas para projetos de ciência de dados e MLOps:.

        ├── dashboard/
        │   └── dashboard.ipynb          # Notebook Python com o dashboard de visualização.
        ├── data/
        │   ├── out/                     # Saídas do DuckDB (tabelas processadas em formato Parquet).
        │   └── src/                     # Dados brutos de entrada (arquivos CSV).
        ├── models/
        │   └── artifacts/               # Artefatos do MLflow (logs de runs, parâmetros, métricas, DataFrames finais).
        ├── notebooks/
        │   └── development_notebook.ipynb # Notebook de desenvolvimento com o processo exploratório completo:
        │                                  # - Processamento manual (tratamento de dados ausentes/inconsistentes).
        │                                  # - Criação de tabelas normalizadas no DuckDB.
        │                                  # - Análise exploratória geral e temporal.
        │                                  # - Feature Engineering com SQL.
        │                                  # - Treinamento de pelo menos dois modelos preditivos (inicialmente).
        │                                  # - Validação temporal e avaliação dos resultados.
        ├── pipeline/                    # Contém o código modularizado da pipeline de ML.
        │   ├── src/
        │   │   ├── utils.py             # Funções auxiliares (cálculo de métricas, criação de offsets de tempo).
        │   │   ├── data_processing.py   # Funções para ingestão de dados e feature engineering com DuckDB.
        │   │   ├── model_training.py    # Lógica de treinamento do modelo LightGBM com validação walk-forward.
        │   │   ├── model_inference.py   # Funções para realizar previsões.
        │   │   └── model_loading.py     # Funções para carregar modelos e artefatos do MLflow.
        │   ├── main_pipeline.py         # Script principal que orquestra todas as etapas da pipeline.
        │   └── requirements.txt         # Lista de dependências Python do projeto.
        └── env_meu_projeto/       # Ambiente virtual Python para o projeto.



**Pré-requisitos**
🛠️Certifique-se de ter o seguinte instalado em seu sistema:
- Python 3.9.13 (ou a versão 3.11.x mais próxima disponível via `pyenv`)
- Git (para Windows, instale o Git Bash para uma melhor experiência de terminal)
- Ferramentas de Build do Visual Studio (para usuários Windows, necessário para compilar pacotes Python como NumPy, SciPy, LightGBM, DuckDB, etc.)
- Conexão com a Internet (para baixar as bibliotecas e o Python via `pyenv`)

**Configuração e Execução 🚀**
Siga os passos abaixo para configurar o ambiente e executar a pipeline.


**Configuração Inicial (Uma Vez)**
---------------------------------
No Windows (usando PowerShell ou Git Bash)

1. Instale o Git para Windows (inclui Git Bash):
Baixe e execute o instalador do Git para Windows: https://git-scm.com/download/win. Durante a instalação, mantenha as opções padrão, incluindo "Git Bash Here".

2. Instale as Ferramentas de Build do Visual Studio:

- Vá para o site oficial da Microsoft: https://visualstudio.microsoft.com/pt-br/downloads/

- Role para baixo até a seção "Ferramentas para Visual Studio" (Tools for Visual Studio).

- Encontre "Build Tools for Visual Studio 2022" e clique em "Baixar" (Download).

- Execute o instalador (`vs_buildtools.exe`).

- No "Instalador do Visual Studio", selecione a carga de trabalho "Desenvolvimento para desktop com C++" (Desktop development with C++).

- No painel direito, em "Detalhes da instalação", verifique se as "Ferramentas de build do MSVC v143" (ou a versão mais recente) e o "SDK do Windows" estão selecionados.

- Clique em "Instalar".

- Reinicie o computador após a instalação.

3. Instale o `pyenv-win`:
- Abra o PowerShell como Administrador.
- Altere a política de execução (se necessário):

        Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        
- Instale o `pyenv-win`:

        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "$env:TEMP\install-pyenv-win.ps1"; & "$env:TEMP\install-pyenv-win.ps1"
- Feche e reabra TODAS as janelas do PowerShell/Git Bash após a instalação para que as variáveis de ambiente sejam carregadas.
- Verifique o PATH: Certifique-se de que `C:\Users\SeuUsuario\.pyenv\pyenv-win\shims` e `C:\Users\SeuUsuario\.pyenv\pyenv-win\bin` 
estejam no TOPO da sua variável de ambiente Path de usuário. Se não estiverem, ajuste-os manualmente nas "Variáveis de Ambiente do Sistema" e reinicie o computador.

4. Instale o Python 3.11.9 com `pyenv`:
- Abra uma nova janela do PowerShell (ou Git Bash).
- Execute:
        pyenv install 3.9.13
- Verifique se foi instalado: `pyenv versions` (deve listar `3.11.9`).
    
    
No Linux / WSL (usando Terminal)

1. Instale o `pyenv`:

        curl https://pyenv.run | bash
Siga as instruções na tela para adicionar `pyenv` ao seu `~/.bashrc` (ou `~/.zshrc`). Feche e reabra o terminal após a configuração.

2. Instale as dependências de compilação do Python:

        sudo apt update
        sudo apt install build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev curl \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

3. Instale o Python 3.9.13 com `pyenv`:

        pyenv install 3.11.9
Verifique se foi instalado: `pyenv versions` (deve listar `3.11.9`).


**Resolução de Conflitos de Dependência (ao usar o Windows)**
--------------------------------------------
Ao tentar instalar as dependências do projeto, surgiram conflitos de compatibilidade entre os pacotes, mesmo após a criação de um ambiente virtual padrão. A causa provável foi um cache de pacotes corrompido ou desatualizado e versões incompatíveis de ferramentas de gerenciamento.

Ocorrendo estes conflitos, a solução foi obtida através destes passos:

- Passo 1: Excluir o Ambiente Virtual Anterior (se existir)

Para evitar qualquer conflito, remova o ambiente virtual anterior (se houver um). Certifique-se de que não há informações importantes que você queira manter nele.

        conda env remove --name [nome_do_ambiente_antigo]

- Passo 2: Criar um Novo Ambiente com Conda

Crie um novo ambiente virtual usando o Conda para garantir o isolamento total das dependências do sistema.

                conda create --name meu-ambiente python=3.9 -y

- Passo 3: Ativar o Novo Ambiente

Ative o ambiente recém-criado para que as próximas instalações ocorram dentro dele.

                conda activate meu-ambiente

- Passo 4: Atualizar o `pip`

Dentro do novo ambiente, atualize o `pip` para a versão mais recente. Isso previne que versões antigas ou corrompidas sejam reutilizadas.

                python -m pip install --upgrade pip

- Passo 5: Limpar cache

                pip cache purge

- Passo 6: Instalar as Dependências do Projeto

Agora, instale as dependências a partir do arquivo `requirements.txt`. Como o ambiente está limpo e o `pip` atualizado, a instalação deve ocorrer sem problemas.

                pip install -r requirements.txt


**Execução da Pipeline**
-------------------------------
**Passo a Passo para Ambos os Sistemas**

1. Navegue até o diretório raiz do projeto:

Este é o diretório que contém as pastas `dashboard/`, `data/`, `models/`, `notebooks/` e `pipeline/`.

        cd /caminho/para/seu/projeto/principal
        # Exemplo Windows: cd C:\Users\SeuUsuario\Documents\MeuProjeto
        # Exemplo Linux/WSL: cd ~/MeuProjeto

2. Defina a versão local do Python para o projeto:
Isso garante que o `pyenv` use o Python 3.11.9 para este diretório e todos os seus subdiretórios.

        pyenv local 3.11.9

Verifique: `python --version` (deve retornar Python `3.11.9`). Se não, revise a configuração do pyenv e do `PATH`.

3. Crie o ambiente virtual (na raiz do projeto):

        python -m venv env_meu_projeto

Isso criará uma pasta `env_meu_projeto` dentro do seu diretório raiz.

4. Ative o ambiente virtual:
- No Windows (PowerShell):     

        .\env_meu_projeto\Scripts\activate

- No Linux / WSL (Bash):

        source env_meu_projeto/bin/activate

Você verá `(env_meu_projeto)` no início da sua linha de comando.

5. Instale as dependências:

Com o ambiente ativado, instale todas as bibliotecas do `requirements.txt`.

        pip install -r pipeline/requirements.txt

6. Execute a Pipeline:

Agora você pode executar o script principal da pipeline, utilizando os argumentos de linha de comando para controlar seu comportamento. Note que o script `main_pipeline.py` está dentro da pasta `pipeline/`.

- Para executar apenas a inferência (comportamento padrão):

        python  pipeline/main_pipeline.py

    (Usará o modelo mais recente logado no MLflow e plotará para o cliente padrão 'C0088').

- Para executar o treinamento completo (ingestão, FE, treino e inferência):

        python  pipeline/main_pipeline.py --train

- Para executar a inferência para um cliente específico:

        python  pipeline/main_pipeline.py --client_id C0042

    (O plot será exibido para o cliente C0042).


- Logs no Terminal: Ao executar o script, você verá as mensagens de log no terminal, indicando o progresso de cada etapa da pipeline (ingestão, feature engineering, treinamento, inferência).
- Artefatos do MLflow: Os logs, parâmetros, métricas e DataFrames finais serão salvos na pasta `models/artifacts/`.

**Detalhes da Pipeline (pipeline/) ⚙️**

A pasta `pipeline/` contém o código modularizado que orquestra as etapas de Machine Learning
- `src/`: Contém os módulos Python com as funções específicas para cada etapa:
    - `data_processing.py`: Lida com a ingestão dos dados brutos (`data/src/`) e a engenharia de features usando **DuckDB**. As tabelas processadas (calendário, features, dados_lgbm_global) são salvas em formato Parquet na pasta `data/out/`.
    - `model_training.py`: Implementa o treinamento do modelo **LightGBM** utilizando validação walk-forward. Os modelos treinados em cada fold e as métricas são rastreados e salvos no MLflow, com os artefatos (incluindo DataFrames finais de treino/validação/teste) armazenados em `models/artifacts/`.
    - `model_inference.py`: Realiza previsões usando o modelo treinado, aplicando as mesmas transformações de pré-processamento para garantir consistência.
    - `model_loading.py`: Permite carregar modelos, DataFrames e parâmetros de runs específicos do MLflow, facilitando a reprodução e a inferência sem a necessidade de retreinar.
    - `utils.py`: Contém funções auxiliares genéricas, como cálculo de métricas de avaliação e manipulação de offsets de tempo.
- `main_pipeline.py`: Atua como o orquestrador principal. Ele importa as funções dos módulos em src/ e as executa em sequência, controlando o fluxo da pipeline (seja para treinamento completo ou apenas inferência/carregamento). Ele também configura o MLflow Tracking URI para apontar para models/artifacts/. Aceita argumentos de linha de comando (--train, --client_id).

Este design modular melhora a manutenibilidade, reusabilidade, testabilidade e a reprodutibilidade do seu projeto de ML.

**Requisitos do Projeto (`requirements.txt`) 📋**

        # requirements.txt
        # Este arquivo lista as dependências Python para o seu projeto.
        # As versões foram escolhidas para garantir compatibilidade, especialmente no Google Colab.

        dash==2.17.0
        duckdb==0.10.2 # Versão atualizada para resolver o problema de instalação
        jupyter==1.0.0
        lightgbm==4.3.0
        matplotlib==3.8.4
        mlflow==2.13.0
        numpy==1.26.4
        packaging==23.2 # Adicionado para resolver o conflito com mlflow
        pandas==2.2.2
        plotly==5.20.0
        pymannkendall==1.4.2
        scikit-learn==1.3.2
        scipy==1.13.0
        seaborn==0.13.2
        sktime==0.28.0
        statsmodels==0.14.2
        ipykernel==6.29.4 # Adicionado para suporte a notebooks Jupyter/VS Code