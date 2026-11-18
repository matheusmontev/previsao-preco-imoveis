"""Preparação dos dados: carregar, limpar e dividir em treino e teste.

Este arquivo é usado pelo treino e pelo site, para que os dois
preparem os dados exatamente do mesmo jeito.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
    """Limpeza que não depende da divisão treino/teste.

    A remoção de outliers fica de fora de propósito: ela usa os limites
    calculados só com o treino (veja remover_extremos_preco_m2).
    """
    dados = df.copy()

    # 1) Só vendas: o tema é preço de venda, e aluguel tem outra escala de preço.
    dados = dados[dados["Negotiation Type"] == "sale"]

    # 2) Anúncios duplicados: linhas idênticas em TODAS as colunas (inclusive
    #    condomínio, vagas e coordenadas) são o mesmo anúncio repetido.
    #    Isso é feito antes de descartar colunas, senão apartamentos diferentes
    #    do mesmo prédio (mesma área e preço) seriam apagados por engano.
    dados = dados.drop_duplicates()

    # 3) Fica só com as colunas que o modelo usa.
    dados = dados[ENTRADAS + [ALVO]]

    # 4) Valores ausentes: remove linhas sem área, quartos, bairro ou preço.
    #    Hoje o dataset não tem nenhum, mas a verificação protege contra dados novos.
    dados = dados.dropna()

    # 5) Valores impossíveis: área, quartos e preço precisam ser positivos.
    dados = dados[(dados["Size"] > 0) & (dados["Rooms"] > 0) & (dados["Price"] > 0)]

    # 6) Transformação do texto: tira o '/São Paulo' do nome do bairro.
    dados["District"] = dados["District"].apply(limpar_bairro)

    return dados.reset_index(drop=True)


def grupos_imovel(X, y):
    """Número do 'grupo' de cada linha: imóveis com mesma área, quartos, bairro
    e preço ficam no mesmo grupo (provavelmente unidades iguais do mesmo prédio).

    Usado para que o mesmo tipo de imóvel não apareça no treino e no teste ao
    mesmo tempo, o que deixaria o erro medido menor do que o real.
    """
    return X.assign(**{ALVO: y}).groupby(ENTRADAS + [ALVO]).ngroup()


def limites_preco_m2(X_treino, y_treino):
    """Limites (1% e 99%) do preço por m², calculados SÓ com o treino."""
    preco_m2 = y_treino / X_treino["Size"]
    baixo, alto = preco_m2.quantile([0.01, 0.99])
    return float(baixo), float(alto)


def remover_extremos_preco_m2(X, y, baixo, alto):
    """Remove imóveis com preço por m² fora de [baixo, alto] (erros de cadastro,
    como R$ 46 mil/m² ou R$ 756/m²)."""
    mantem = (y / X["Size"]).between(baixo, alto)
    return X[mantem], y[mantem]


def dividir_treino_teste(dados, tamanho_teste=0.2, semente=42):
    """Separa 80% para treino e 20% para teste, sorteando por grupo de imóvel.

    random_state (semente) fixa o sorteio: rodando de novo, a divisão é a mesma.
    Depois da divisão, os extremos de preço por m² são removidos de treino e
    teste usando limites calculados só com o treino.
    """
    X = dados[ENTRADAS]
    y = dados[ALVO]
    grupos = grupos_imovel(X, y)

    divisor = GroupShuffleSplit(n_splits=1, test_size=tamanho_teste, random_state=semente)
    idx_treino, idx_teste = next(divisor.split(X, y, groups=grupos))
    X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]
    y_treino, y_teste = y.iloc[idx_treino], y.iloc[idx_teste]

    baixo, alto = limites_preco_m2(X_treino, y_treino)
    X_treino, y_treino = remover_extremos_preco_m2(X_treino, y_treino, baixo, alto)
    X_teste, y_teste = remover_extremos_preco_m2(X_teste, y_teste, baixo, alto)
    # Devolve também os limites usados: não dá para recalculá-los depois,
    # porque o treino já está sem os extremos.
    return X_treino, X_teste, y_treino, y_teste, (baixo, alto)


if __name__ == "__main__":
    # Roda só quando o arquivo é executado direto (python preparacao.py),
    # não quando é importado pelo treino ou pelo site.
    df = carregar_dados()
    vendas = df[df["Negotiation Type"] == "sale"]
    dados = preparar_dados(df)
    print("Linhas originais:", len(df))
    print("Vendas:", len(vendas))
    print("Duplicatas removidas (idênticas em todas as colunas):", len(vendas) - len(vendas.drop_duplicates()))
    print("Linhas após a preparação:", len(dados))
    print("Bairros:", dados["District"].nunique())
    print("\nExemplo dos dados prontos:")
    print(dados.head())

    X_treino, X_teste, y_treino, y_teste, (baixo, alto) = dividir_treino_teste(dados)
    print(f"\nPreço/m² mantido entre R$ {baixo:,.0f} e R$ {alto:,.0f} (limites do treino)")
    print("Treino:", len(X_treino), "imóveis")
    print("Teste: ", len(X_teste), "imóveis")

    # Confere que nenhum imóvel igual aparece no treino e no teste ao mesmo tempo.
    iguais = set(map(tuple, X_treino.assign(Price=y_treino).values)) & set(
        map(tuple, X_teste.assign(Price=y_teste).values))
    print("Imóveis iguais em treino e teste ao mesmo tempo:", len(iguais))

    faltando = set(X_teste["District"]) - set(X_treino["District"])
    print("Bairros do teste que não aparecem no treino:", faltando or "nenhum")
    poucos = dados["District"].value_counts()
    print("Bairros com menos de 10 anúncios:", (poucos < 10).sum())
