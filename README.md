# Estimador de preço de imóveis em São Paulo

Site que estima o preço de venda de um imóvel em São Paulo a partir da **área**, do número de **quartos** e do **bairro**, usando uma **Árvore de Regressão**. Trabalho de Mineração de Dados.

- **Site (link público):** https://previsao-preco-imoveis-tcnjprszjesrrjoa8quthy.streamlit.app/
- **Integrantes:** Matheus Silva do Monte e Gustavo Azevedo Diniz
- **Professora:** Bruna Silva Celestino

## O que o site faz

O usuário informa área, quartos e bairro e recebe o **preço estimado** e uma **faixa de erro**. Também dá para ver o que o outro modelo (preço total) estimaria, para comparar. A aba "Como funciona" explica a técnica, os dados e as métricas. O site valida as entradas (área e quartos dentro do que o modelo conhece, bairro escolhido numa lista) e avisa quando a combinação pedida está fora do que existe nos dados, como um imóvel grande num bairro que só tem apartamentos pequenos.

## Dados

- **Fonte:** Kaggle, [*São Paulo Real Estate - Sale / Rent - April 2019*](https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019) (arquivo `sao-paulo-properties-april-2019.csv`). Dados **reais** de anúncios de abril de 2019, **não simulados**. Confira a licença na página do dataset antes de redistribuir.
- 13.640 anúncios (venda e aluguel). O projeto usa só os 6.412 de **venda**.
- Arquivo no repositório: `dados/sao-paulo-properties-april-2019.csv`.

## Preparação dos dados (`preparacao.py`)

1. Mantém só anúncios de venda.
2. Remove 110 anúncios duplicados (idênticos em todas as colunas).
3. Mantém apenas área, quartos, bairro e preço.
4. Remove linhas com valores ausentes ou impossíveis (área, quartos ou preço menores ou iguais a zero).
5. Limpa o nome do bairro (`Moema/São Paulo` vira `Moema`).
6. Divide em 80% treino e 20% teste, sorteando por **grupo de imóvel** (mesma área, quartos, bairro e preço) para que unidades iguais não caiam no treino e no teste ao mesmo tempo.
7. Remove o 1% mais baixo e o 1% mais alto do preço por m² (prováveis erros de cadastro), com limites calculados **só com o treino**.

O bairro é transformado em colunas de 0 e 1 (One-Hot Encoding) dentro do modelo. Não há normalização, porque a árvore divide os dados por limiares e a escala não muda as divisões.

## Modelos e resultados

Treinamos **dois modelos**, os dois com `DecisionTreeRegressor` dentro de um `Pipeline` (One-Hot do bairro + árvore) e as mesmas entradas. Muda só o que a árvore aprende a prever:

| Modelo | A árvore prevê | Preço final | Arquivo |
|---|---|---|---|
| **Preço por m²** (principal, usado no site) | o valor do m² | `preço por m² × área` | `modelo/modelo_arvore_m2.joblib` |
| **Preço total** (comparação) | o preço inteiro | a própria previsão | `modelo/modelo_arvore_preco.joblib` |

**Por que preço por m²:** o preço total mistura tamanho e localização. Prevendo o preço por m², a parte proporcional ao tamanho fica com a multiplicação, e as perguntas da árvore sobram para separar os bairros. O preço por m² não é único para a cidade: cada bairro tem o seu, e a área e os quartos continuam como entradas.

Os parâmetros (`max_depth`, `min_samples_leaf`) de cada modelo foram escolhidos por validação cruzada com `GroupKFold` (5 partes) apenas no treino, usando o erro em reais do preço final nos dois casos.

| Métrica (1.224 imóveis de teste) | Preço por m² | Preço total | Referência (sempre a média) |
|---|---|---|---|
| Erro médio absoluto (MAE) | R$ 108.458 | R$ 129.418 | R$ 346.911 |
| R² | 0,87 | 0,71 | 0,00 |
| Faixa de erro (80% dos casos) | ±27,4% | ±29,6% | |

Como um único sorteio de treino/teste pode enganar, repetimos tudo com **10 sorteios diferentes**. O modelo por m² errou menos **em todos os 10**: MAE médio de R$ 112 mil contra R$ 126 mil (cerca de 11% menos) e R² médio de 0,84 contra 0,78.

## Como rodar

Testado com Python 3.15.2 e as versões fixadas no `requirements.txt`.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

O site abre em `http://localhost:8501`. O `app.py` não funciona com `python app.py` nem com o botão ▶ do VS Code: o Streamlit precisa ser iniciado pelo próprio comando.

Os modelos já vêm treinados em `modelo/`. Para treinar de novo, execute `treino_modelo.ipynb` (ele regrava os dois `.joblib` e o `modelo/metadados.json`; leva alguns minutos por causa da comparação em 10 sorteios). Depois, `python 03_desenhar_arvore.py` atualiza os desenhos das árvores.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `app.py` | O site em Streamlit |
| `preparacao.py` | Limpeza e divisão treino/teste (usada pelo notebook; o site recebe os bairros já limpos pelo `modelo/metadados.json`) |
| `treino_modelo.ipynb` | Treino dos dois modelos, escolha de parâmetros, métricas, comparação e salvamento |
| `01_exploracao.py`, `02_precos_vendas.py` | Exploração dos dados |
| `03_desenhar_arvore.py` | Desenha o topo das duas árvores em `imagens/` |
| `modelo/` | Os dois modelos salvos (joblib) e as informações usadas pelo site |
| `imagens/` | Desenhos das árvores |
| `dados/` | Dataset |

## Limitações

- Os preços são de abril de 2019.
- Só vale para São Paulo: o dataset não tem outras cidades.
- O modelo só conhece área, quartos e bairro (não considera idade, andar, vagas, condomínio, acabamento).
- A árvore não extrapola: em combinações que não existem nos dados, a estimativa é menos confiável (o site avisa).
- A faixa de erro é a mesma em porcentagem para todos os imóveis.

## Uso de IA

Usamos o **Claude Code** (Anthropic) como assistente durante o desenvolvimento, para:

- explicar os conceitos (árvore de regressão, One-Hot Encoding, validação cruzada, métricas);
- documentar o código de `preparacao.py` e ajudar na limpeza (duplicatas, divisão por grupo, extremos calculados só com o treino);
- escrever, organizar e documentar o notebook de treino e o site (`app.py`), fazer o script que desenha as árvores e tirar nossas dúvidas;
- investigar problemas nos dados dos quais a dupla desconfiava (duplicatas, valores extremos), verificar e revisar o código a nosso pedido e apontar a versão do Python incompatível;
- comparar o modelo de preço total com o de preço por m², que adotamos depois de ver o resultado em 10 sorteios;
- escrever testes automáticos simples do site e este README.

As escolhas do projeto (dataset, filtrar só vendas, tratamento dos extremos de preço, divisão 80/20, usar o modelo por m² e manter o de preço total para comparação) foram discutidas e decididas pela dupla.

**Commits:** a dupla decidia quando fazer cada commit e pedia mensagens semânticas (`feat:`, `fix:`, `docs:`). O Claude Code escrevia a mensagem e verificava o código antes (rodando o notebook e testes do site). O commit deste README é assinado pelo Claude e traz na mensagem o pedido feito a ele.

**Segurança:** verificamos os arquivos e todo o histórico do git: não há senha, chave de API ou token no repositório. Os únicos dados pessoais são os nomes da dupla e da professora, no início deste README.
