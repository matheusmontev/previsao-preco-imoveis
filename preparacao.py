"""Preparação dos dados: carregar, limpar e dividir em treino e teste.

Este arquivo é usado pelo treino e pelo site, para que os dois
preparem os dados exatamente do mesmo jeito.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_CSV = PASTA_PROJETO / "dados" / "sao-paulo-properties-april-2019.csv"

# Colunas que o usuário informa no site (entradas) e a que queremos prever (alvo).
ENTRADAS = ["Size", "Rooms", "District"]
ALVO = "Price"


def carregar_dados(caminho=CAMINHO_CSV):
    """Lê o CSV original, sem nenhuma alteração."""
    return pd.read_csv(caminho, encoding="utf-8")


def limpar_bairro(nome):
    """'Moema/São Paulo' -> 'Moema'. Todos os bairros são de São Paulo."""
    return nome.split("/")[0].strip()


def preparar_dados(df):
    """Aplica todas as etapas de limpeza e devolve uma tabela pronta para o modelo."""
    dados = df.copy()

    # 1) Só vendas: o tema é preço de venda, e aluguel tem outra escala de preço.
    dados = dados[dados["Negotiation Type"] == "sale"]

    # 2) Fica só com as colunas que o modelo usa.
    dados = dados[ENTRADAS + [ALVO]]

    # 3) Valores ausentes: remove linhas sem área, quartos, bairro ou preço.
    #    Hoje o dataset não tem nenhum, mas a verificação protege contra dados novos.
    dados = dados.dropna()

    # 4) Valores impossíveis: área, quartos e preço precisam ser positivos.
    dados = dados[(dados["Size"] > 0) & (dados["Rooms"] > 0) & (dados["Price"] > 0)]

    # 5) Transformação do texto: tira o '/São Paulo' do nome do bairro.
    dados["District"] = dados["District"].apply(limpar_bairro)

    # 6) Outliers: remove 1% mais baixo e 1% mais alto do preço por m²,
    #    que indicam erro de cadastro (ex.: R$ 46 mil/m² ou R$ 756/m²).
    preco_m2 = dados["Price"] / dados["Size"]
    baixo, alto = preco_m2.quantile([0.01, 0.99])
    dados = dados[preco_m2.between(baixo, alto)]

    return dados.reset_index(drop=True)


def dividir_treino_teste(dados, tamanho_teste=0.2, semente=42):
    """Separa 80% para treino e 20% para teste.

    random_state (semente) fixa o sorteio: rodando de novo, a divisão é a mesma.
    """
    X = dados[ENTRADAS]
    y = dados[ALVO]
    return train_test_split(X, y, test_size=tamanho_teste, random_state=semente)


if __name__ == "__main__":
    # Roda só quando o arquivo é executado direto (python preparacao.py),
    # não quando é importado pelo treino ou pelo site.
    df = carregar_dados()
    dados = preparar_dados(df)
    print("Linhas originais:", len(df))
    print("Linhas após a preparação:", len(dados))
    print("Bairros:", dados["District"].nunique())
    print("\nExemplo dos dados prontos:")
    print(dados.head())

    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(dados)
    print("\nTreino:", len(X_treino), "imóveis")
    print("Teste: ", len(X_teste), "imóveis")

    # Bairros com poucos anúncios podem não aparecer no treino.
    poucos = dados["District"].value_counts()
    print("\nBairros com menos de 10 anúncios:", (poucos < 10).sum())
    faltando = set(X_teste["District"]) - set(X_treino["District"])
    print("Bairros do teste que não aparecem no treino:", faltando or "nenhum")
