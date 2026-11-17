import pandas as pd
from pathlib import Path

#o CSV fica dentro do projeto, em dados/
pasta_projeto = Path(__file__).resolve().parent
caminho = pasta_projeto / "dados" / "sao-paulo-properties-april-2019.csv"
df = pd.read_csv(caminho, encoding="utf-8")

# Filtra só as vendas. O dataset mistura venda ('sale') e aluguel ('rent').
vendas = df[df["Negotiation Type"] == "sale"].copy()
print("Total de imóveis:", len(df))
print("Só vendas:", len(vendas))

# Estatísticas básicas do preço das vendas.
# count = quantidade, mean = média, 50% = mediana, min/max = extremos.
print("\nEstatísticas do preço (R$):")
print(vendas["Price"].describe().round(0))

# Mediana x média: a média é puxada pelos valores muito altos.
# Se a média for bem maior que a mediana, há imóveis caros demais puxando a conta.
print("\nMediana do preço:", vendas["Price"].median())
print("Média do preço:  ", round(vendas["Price"].mean()))

# Percentis: o valor abaixo do qual está essa fração dos imóveis.
# 0.99 = 99% dos imóveis custam menos que esse valor.
print("\nPercentis do preço:")
print(vendas["Price"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

# Relação entre área e preço: correlação perto de 1 = quanto maior a área, maior o preço.
print("\nCorrelação área x preço:", round(vendas["Size"].corr(vendas["Price"]), 3))

# Outliers pelo método do IQR (intervalo interquartil):
# valores abaixo de Q1 - 1,5*IQR ou acima de Q3 + 1,5*IQR são suspeitos.
q1 = vendas["Price"].quantile(0.25)
q3 = vendas["Price"].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
outliers = vendas[(vendas["Price"] < limite_inferior) | (vendas["Price"] > limite_superior)]
print(f"\nLimite inferior: R$ {limite_inferior:,.0f}")
print(f"Limite superior: R$ {limite_superior:,.0f}")
print("Imóveis fora desse intervalo:", len(outliers))

# ---------------------------------------------------------------
# O que explica mais o preço: a área, os quartos ou o bairro?
# ---------------------------------------------------------------
# R² = fração da variação do preço que uma variável "explica" (de 0 a 1).
# - Área e quartos são números: R² = correlação ao quadrado.
# - Bairro é texto: usamos a média de preço de cada bairro como "previsão"
#   e medimos quanto da variação ela explica.
def r2_por_grupo(dados, coluna):
    media_do_grupo = dados.groupby(coluna)["Price"].transform("mean")
    variacao_total = ((dados["Price"] - dados["Price"].mean()) ** 2).sum()
    variacao_restante = ((dados["Price"] - media_do_grupo) ** 2).sum()
    return 1 - variacao_restante / variacao_total

print("\nQuanto cada variável explica do preço (R²):")
print("  Área:    ", round(vendas["Size"].corr(vendas["Price"]) ** 2, 3))
print("  Quartos: ", round(vendas["Rooms"].corr(vendas["Price"]) ** 2, 3))
print("  Bairro:  ", round(r2_por_grupo(vendas, "District"), 3))

# Preço por m² tira o efeito do tamanho e mostra só o "peso" do bairro.
vendas["preco_m2"] = vendas["Price"] / vendas["Size"]
por_bairro = vendas.groupby("District")["preco_m2"].median().sort_values()
print("\n5 bairros mais baratos (mediana R$/m²):")
print(por_bairro.head(5).round(0))
print("\n5 bairros mais caros (mediana R$/m²):")
print(por_bairro.tail(5).round(0))

# ---------------------------------------------------------------
# Limpeza escolhida: remover 1% mais baixo e 1% mais alto do preço por m²
# ---------------------------------------------------------------
# Esses extremos indicam erro de cadastro (ex.: R$ 46 mil/m² ou R$ 756/m²).
# Os imóveis caros "normais" continuam, porque são reais.
limite_m2_baixo = vendas["preco_m2"].quantile(0.01)
limite_m2_alto = vendas["preco_m2"].quantile(0.99)
vendas_limpas = vendas[vendas["preco_m2"].between(limite_m2_baixo, limite_m2_alto)]
print(f"\nPreço/m² mantido entre R$ {limite_m2_baixo:,.0f} e R$ {limite_m2_alto:,.0f}")
print("Removidos:", len(vendas) - len(vendas_limpas), "| Restantes:", len(vendas_limpas))
