"""Site: Estimador de preço de imóveis em São Paulo (Árvore de Regressão).

Como rodar:  python -m streamlit run app.py
Os modelos já vêm treinados (pasta modelo/); o site não treina nada ao abrir.

Há dois modelos salvos, treinados no notebook treino_modelo.ipynb:
- "m2" (principal): a árvore prevê o preço por m², e o site multiplica pela área.
- "preco" (comparação): a árvore prevê o preço total direto.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PASTA_MODELO = Path(__file__).resolve().parent / "modelo"

st.set_page_config(page_title="Estimador de preço de imóveis", page_icon="🏠")


@st.cache_resource
def carregar_modelos():
    """Carrega os dois modelos e as informações deles uma única vez (não a cada clique)."""
    with open(PASTA_MODELO / "metadados.json", encoding="utf-8") as f:
        meta = json.load(f)
    modelos = {
        alvo: joblib.load(PASTA_MODELO / info["arquivo"])
        for alvo, info in meta["modelos"].items()
    }
    return modelos, meta


def milhar(numero):
    """1234567 -> '1.234.567' (ponto como separador de milhar, como no Brasil)."""
    return f"{numero:,.0f}".replace(",", ".")


def reais(valor):
    """1234567 -> 'R$ 1.234.567'."""
    return "R$ " + milhar(valor)


def validar(area, quartos, bairro, meta):
    """Devolve a lista de problemas encontrados nas entradas (vazia = tudo certo)."""
    erros = []
    if area is None or area != area:  # None ou NaN
        erros.append("Informe a área do imóvel.")
    elif not meta["area_min"] <= area <= meta["area_max"]:
        erros.append(f"A área deve estar entre {meta['area_min']} e {meta['area_max']} m².")
    if quartos is None or not meta["quartos_min"] <= quartos <= meta["quartos_max"]:
        erros.append(f"O número de quartos deve estar entre {meta['quartos_min']} e {meta['quartos_max']}.")
    if bairro not in meta["bairros"]:
        erros.append("Escolha um bairro da lista.")
    return erros


def estimar(modelo, info, area, quartos, bairro):
    """Faz a previsão em reais e monta a faixa de erro.

    No modelo 'm2' a árvore devolve o preço por m², então multiplicamos pela área.
    Devolve (estimativa, minimo, maximo), arredondados para o milhar.
    """
    entrada = pd.DataFrame([{"Size": area, "Rooms": quartos, "District": bairro}])
    previsto = float(modelo.predict(entrada)[0])
    estimativa = previsto * area if info["alvo"] == "m2" else previsto
    faixa = info["faixa_pct"]
    return (
        round(estimativa, -3),
        round(estimativa * (1 - faixa), -3),
        round(estimativa * (1 + faixa), -3),
    )


try:
    modelos, meta = carregar_modelos()
except FileNotFoundError:
    st.error("Modelo não encontrado. Rode o notebook `treino_modelo.ipynb` para gerar a pasta `modelo/`.")
    st.stop()

PRINCIPAL = meta["modelo_principal"]           # "m2"
OUTRO = "preco" if PRINCIPAL == "m2" else "m2"  # modelo usado só para comparação
info = meta["modelos"][PRINCIPAL]
info_outro = meta["modelos"][OUTRO]

st.title("🏠 Estimador de preço de imóveis")
st.write("Descubra quanto custa, aproximadamente, um imóvel à venda em São Paulo.")

aba_estimar, aba_como = st.tabs(["Estimar preço", "Como funciona"])

# ----------------------------------------------------------------------------
# Aba 1: formulário e resposta
# ----------------------------------------------------------------------------
with aba_estimar:
    with st.form("formulario"):
        area = st.number_input(
            "Área (m²)",
            min_value=float(meta["area_min"]),
            max_value=float(meta["area_max"]),
            value=70.0,
            step=1.0,
            help=f"Entre {meta['area_min']} e {meta['area_max']} m².",
        )
        quartos = st.selectbox(
            "Quartos",
            options=list(range(meta["quartos_min"], meta["quartos_max"] + 1)),
            index=1,
        )
        bairro = st.selectbox(
            "Bairro",
            options=meta["bairros"],
            index=None,
            placeholder="Digite ou escolha o bairro",
        )
        enviado = st.form_submit_button("Estimar preço", type="primary")

    if enviado:
        erros = validar(area, quartos, bairro, meta)
        if erros:
            for erro in erros:
                st.error(erro)
        else:
            # Aviso: a árvore não extrapola. Se o bairro não tem anúncios com essa
            # área, a estimativa vem de imóveis de outros bairros.
            area_min_b, area_max_b, qtd_b = meta["area_por_bairro"][bairro]
            if not area_min_b <= area <= area_max_b:
                st.warning(
                    f"Em {bairro}, o modelo conhece {qtd_b} anúncios, com áreas de "
                    f"{area_min_b} a {area_max_b} m². Para {area:.0f} m² a estimativa é "
                    f"**menos confiável**, porque se parece mais com imóveis de outros bairros."
                )

            estimativa, minimo, maximo = estimar(modelos[PRINCIPAL], info, area, quartos, bairro)

            st.subheader("Resultado")
            st.metric("Preço estimado", reais(estimativa))
            col1, col2 = st.columns(2)
            col1.metric("Faixa: mínimo", reais(minimo))
            col2.metric("Faixa: máximo", reais(maximo))
            st.write(
                f"Um imóvel de **{area:.0f} m²** com **{quartos} quarto(s)** em **{bairro}** "
                f"deve custar perto de **{reais(estimativa)}** "
                f"(cerca de {reais(estimativa / area)} por m²)."
            )
            st.caption(
                f"Nos imóveis que o modelo não conhecia, o preço real ficou dentro de uma faixa "
                f"de ±{info['faixa_pct']:.0%} da estimativa em 8 a cada 10 casos. "
                f"É uma estimativa baseada em anúncios de abril de 2019, não uma avaliação oficial."
            )

            with st.expander(f"Comparar com o outro modelo ({info_outro['nome'].lower()})"):
                outra, _, _ = estimar(modelos[OUTRO], info_outro, area, quartos, bairro)
                st.write(
                    f"O modelo que prevê o **{info_outro['nome'].lower()}** estimaria "
                    f"**{reais(outra)}** para este imóvel. Ele erra mais em média "
                    f"({reais(info_outro['mae_10_sorteios'])} contra {reais(info['mae_10_sorteios'])}), "
                    f"por isso o site usa o modelo de {info['nome'].lower()}. "
                    f"Veja a comparação na aba \"Como funciona\"."
                )

# ----------------------------------------------------------------------------
# Aba 2: como funciona
# ----------------------------------------------------------------------------
with aba_como:
    st.header("Como funciona")

    st.subheader("A técnica: Árvore de Regressão")
    st.write(
        "A árvore aprende fazendo perguntas sobre os imóveis, como *\"o bairro é Moema?\"* "
        "e depois *\"a área é maior que 85 m²?\"*. Cada resposta separa os imóveis em grupos mais "
        "parecidos. Ela segue as perguntas até um grupo final e devolve **a média desse grupo**."
    )
    st.write(
        "A árvore estima o **preço por m²** do imóvel, e o site multiplica pela área:\n\n"
        "**preço estimado = preço por m² previsto × área**\n\n"
        "Assim a árvore não gasta perguntas aprendendo que \"imóvel maior custa mais\" "
        "(a multiplicação já cuida disso) e usa as perguntas para separar as regiões. "
        "O preço por m² **não é o mesmo para a cidade toda**: cada bairro tem o seu, e a área "
        "e os quartos também podem mudá-lo."
    )

    st.subheader("Os dados")
    st.write(
        f"- **Fonte:** Kaggle, *São Paulo Real Estate - Sale / Rent - April 2019*. "
        f"São anúncios **reais** (não simulados), de abril de 2019.\n"
        f"- O arquivo tem {milhar(meta['linhas_dataset'])} anúncios de venda e aluguel; usamos só as "
        f"**{milhar(meta['vendas_dataset'])} vendas**.\n"
        f"- Removemos {meta['duplicatas_removidas']} anúncios repetidos e os imóveis com preço por m² "
        f"fora de {reais(meta['preco_m2_min'])} a {reais(meta['preco_m2_max'])} (provável erro de cadastro).\n"
        f"- Sobraram **{milhar(meta['linhas_limpas'])} imóveis** em {len(meta['bairros'])} bairros."
    )

    st.subheader("Como medimos o acerto")
    st.write(
        f"Separamos **{milhar(meta['linhas_treino'])} imóveis para o modelo aprender** e "
        f"**{milhar(meta['linhas_teste'])} para testar**, imóveis que ele nunca viu."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Erro médio (MAE)", reais(info["mae"]), help="Em média, a estimativa erra este valor para mais ou para menos.")
    c2.metric("R²", f"{info['r2']:.2f}", help="De 0 a 1: quanto da variação dos preços o modelo explica.")
    c3.metric("Faixa de erro", f"±{info['faixa_pct']:.0%}", help="Em 80% dos imóveis de teste, o preço real ficou dentro desta faixa.")

    st.write("**O que mais pesa na estimativa:**")
    st.bar_chart(pd.Series(info["importancias"]).sort_values(ascending=False))

    st.subheader("Por que preço por m²? Comparação dos dois modelos")
    st.write(
        "Treinamos também uma árvore que prevê o **preço total** direto. Para a comparação não "
        "depender da sorte de um sorteio, repetimos a divisão treino/teste **10 vezes** com sorteios "
        f"diferentes. O modelo por m² errou menos em **{meta['vitorias_m2_10_sorteios']} de 10**."
    )
    tabela = pd.DataFrame(
        {
            m["nome"]: {
                "Erro médio (MAE), 10 sorteios": reais(m["mae_10_sorteios"]),
                "R², 10 sorteios": f"{m['r2_10_sorteios']:.2f}".replace(".", ","),
                "Faixa de erro (80% dos casos)": f"±{m['faixa_pct']:.0%}",
                "Peso do bairro": f"{m['importancias']['Bairro']:.0%}",
            }
            for m in (info, info_outro)
        }
    )
    st.table(tabela)

    st.subheader("Limitações")
    st.write(
        "- Os preços são de **abril de 2019**; hoje os valores são mais altos.\n"
        "- Só vale para **São Paulo**: o modelo não conhece outras cidades.\n"
        "- O modelo só conhece área, quartos e bairro. Idade, andar, vagas, condomínio e acabamento "
        "também mudam o preço, mas ficam de fora.\n"
        "- A árvore não extrapola: em combinações que não existem nos dados (como um imóvel grande "
        "num bairro que só tem apartamentos pequenos), a estimativa é menos confiável. O site avisa.\n"
        "- A faixa de erro é a mesma em porcentagem para todos os imóveis; em imóveis raros "
        "(muito grandes ou de alto padrão) o erro tende a ser maior.\n"
        "- É uma estimativa, não uma avaliação oficial de imóvel."
    )
