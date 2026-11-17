import pandas as pd

# Caminho do arquivo CSV
caminho = "dados/sao-paulo-properties-april-2019.csv"

# read_csv lê o arquivo e cria um DataFrame (uma tabela com linhas e colunas).
df = pd.read_csv(caminho, encoding="utf-8")

# Quantas linhas e colunas?
print("Linhas e colunas:", df.shape)

# Quais são as colunas e que tipo de dado cada uma tem?
print(df.dtypes)

# Primeiras 5 linhas, para ver como os dados são
df.head()

# Quantos valores estão faltando em cada coluna?
print(df.isna().sum())