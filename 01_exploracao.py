import pandas as pd
from pathlib import Path

# Caminho do CSV dentro do próprio projeto, relativo a este arquivo.
# Assim o script funciona em qualquer máquina que clonar o repositório.
pasta_projeto = Path(__file__).resolve().parent
caminho = pasta_projeto / "dados" / "sao-paulo-properties-april-2019.csv"

# read_csv lê o arquivo e cria um DataFrame (tabela com linhas e colunas)
df = pd.read_csv(caminho, encoding="utf-8")

# Quantas linhas e colunas?
print("Linhas e colunas:", df.shape)

# Quais colunas e que tipo de dado cada uma tem?
print(df.dtypes)

# Primeiras 5 linhas (print é necessário quando o arquivo roda como script)
print(df.head())

# Quantos valores estão faltando em cada coluna?
print(df.isna().sum())
