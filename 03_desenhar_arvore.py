"""Desenha os primeiros níveis das duas árvores treinadas (pasta modelo/).

A árvore completa tem centenas de folhas e não cabe numa imagem, então o desenho
mostra só o topo, onde estão as perguntas mais importantes.

Como rodar:  python 03_desenhar_arvore.py
Gera:  imagens/arvore_m2.png     (modelo principal: preço por m²)
       imagens/arvore_preco.png  (modelo de comparação: preço total)
"""
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # gera a imagem sem abrir janela
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

PASTA = Path(__file__).resolve().parent
NIVEIS = 3  # profundidade desenhada (0 a 3 = 4 fileiras de perguntas)


def nomes_legiveis(modelo):
    """'bairro__District_Moema' -> 'Bairro Moema'; Size/Rooms -> Área/Quartos."""
    nomes = []
    for nome in modelo.named_steps["preparo"].get_feature_names_out():
        if nome == "remainder__Size":
            nomes.append("Área (m²)")
        elif nome == "remainder__Rooms":
            nomes.append("Quartos")
        else:
            nomes.append("Bairro " + nome.replace("bairro__District_", ""))
    return nomes


def desenhar(modelo, info, saida):
    arvore = modelo.named_steps["arvore"]
    # No modelo por m², o valor de cada caixa é a média do preço por m², não do preço.
    rotulo_valor = "R$/m² médio" if info["alvo"] == "m2" else "preço médio"

    fig, ax = plt.subplots(figsize=(30, 11))
    textos = plot_tree(
        arvore,
        max_depth=NIVEIS,
        feature_names=nomes_legiveis(modelo),
        impurity=False,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )

    # Traduz o texto de cada caixa para português.
    for texto in textos:
        t = texto.get_text()
        # No One-Hot, "Bairro X <= 0.5" é verdadeiro quando o imóvel NÃO é do bairro X.
        # Escrevemos a pergunta de modo que "Sim" (esquerda) continue batendo com a árvore.
        t = re.sub(r"Bairro ([^\n]+?) <= 0\.5", r"Bairro diferente\nde \1?", t)
        t = re.sub(
            r"(Área \(m²\)|Quartos) <= ([\d.]+)",
            lambda m: f"{m.group(1)} ≤ {m.group(2).replace('.', ',')}?",
            t,
        )
        t = re.sub(r"samples = (\d+)", r"imóveis = \1", t)
        t = re.sub(
            r"value = ([\d.]+)",
            lambda m: f"{rotulo_valor} = R$ " + f"{float(m.group(1)):,.0f}".replace(",", "."),
            t,
        )
        texto.set_text(t)
        # Sem isto, o texto entre dois "$" vira fórmula matemática e sai torto.
        texto.set_parse_math(False)

    # Os rótulos dos ramos vêm em inglês (com espaços em volta); troca por Sim/Não.
    for texto in ax.texts:
        if texto.get_text().strip() == "True":
            texto.set_text("Sim")
        elif texto.get_text().strip() == "False":
            texto.set_text("Não")

    ax.set_title(
        f"Árvore de Regressão, modelo de {info['nome'].lower()}: os {NIVEIS + 1} primeiros níveis de perguntas "
        f"(a árvore completa tem {arvore.get_depth()} níveis e {arvore.get_n_leaves()} folhas)\n"
        "Cada caixa é uma pergunta. Sim = ramo da esquerda, Não = ramo da direita. "
        f"Cor mais escura = {rotulo_valor} mais alto.",
        fontsize=13,
    )

    saida.parent.mkdir(exist_ok=True)
    fig.savefig(saida, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("Imagem salva em:", saida)


with open(PASTA / "modelo" / "metadados.json", encoding="utf-8") as f:
    meta = json.load(f)

for alvo, info in meta["modelos"].items():
    modelo = joblib.load(PASTA / "modelo" / info["arquivo"])
    desenhar(modelo, info, PASTA / "imagens" / f"arvore_{alvo}.png")
