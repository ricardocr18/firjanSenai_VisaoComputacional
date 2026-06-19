# =============================================================================
# secao_05_cnn_analise.py
# Desafio 01 SA — Detecção e Classificação de Objetos com OpenCV
# =============================================================================
# Relaciona o pipeline desenvolvido com redes neurais convolucionais (CNNs),
# analisa os resultados obtidos, limitações e possíveis melhorias.
# Gera figura comparativa do pipeline manual vs CNN.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def exibir_pipeline_vs_cnn():
    """
    Gera figura comparando o pipeline manual com uma abordagem CNN.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Pipeline Manual vs Rede Neural Convolucional (CNN)\n"
        "Comparação de Abordagens para Classificação Visual",
        fontsize=13, fontweight="bold",
    )

    # --- Painel 1: Pipeline Manual (este projeto) ---
    ax1 = eixos[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Pipeline Manual (este projeto)", fontsize=11,
                  fontweight="bold", color="#1a5276")

    etapas_manual = [
        (5, 9.2, "#2e86c1", "Imagem de entrada"),
        (5, 7.7, "#1a5276", "Pré-processamento\n(cinza, blur, CLAHE, Otsu)"),
        (5, 6.2, "#1a5276", "Segmentação\n(limiarização + morfologia)"),
        (5, 4.7, "#1a5276", "Detecção de contornos\n(findContours)"),
        (5, 3.2, "#1a5276", "Extração manual\nde características"),
        (5, 1.7, "#27ae60", "Classificação\npor regras (if/else)"),
        (5, 0.4, "#27ae60", "Resultado: classe do objeto"),
    ]

    for x, y, cor, texto in etapas_manual:
        ax1.add_patch(mpatches.FancyBboxPatch(
            (x - 2.8, y - 0.5), 5.6, 0.9,
            boxstyle="round,pad=0.1", facecolor=cor,
            edgecolor="white", linewidth=1.5, alpha=0.9,
        ))
        ax1.text(x, y, texto, ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold")
        if y > 0.6:
            ax1.annotate("", xy=(x, y - 0.62), xytext=(x, y - 0.5),
                         arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))

    ax1.text(5, -0.5,
             "Características: área, aspect ratio,\ncircularidade, solidez, cor média",
             ha="center", fontsize=8, color="#555", style="italic")

    # --- Painel 2: CNN ---
    ax2 = eixos[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Rede Neural Convolucional (CNN)", fontsize=11,
                  fontweight="bold", color="#6e2fa1")

    etapas_cnn = [
        (5, 9.2, "#8e44ad", "Imagem de entrada"),
        (5, 7.7, "#6e2fa1", "Camadas convolucionais\n(extração automática de features)"),
        (5, 6.2, "#6e2fa1", "Pooling\n(redução dimensional)"),
        (5, 4.7, "#6e2fa1", "Mais camadas conv.\n(features de alto nível)"),
        (5, 3.2, "#6e2fa1", "Camadas densas\n(fully connected)"),
        (5, 1.7, "#27ae60", "Softmax\n(probabilidade por classe)"),
        (5, 0.4, "#27ae60", "Resultado: classe + confiança"),
    ]

    for x, y, cor, texto in etapas_cnn:
        ax2.add_patch(mpatches.FancyBboxPatch(
            (x - 2.8, y - 0.5), 5.6, 0.9,
            boxstyle="round,pad=0.1", facecolor=cor,
            edgecolor="white", linewidth=1.5, alpha=0.9,
        ))
        ax2.text(x, y, texto, ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold")
        if y > 0.6:
            ax2.annotate("", xy=(x, y - 0.62), xytext=(x, y - 0.5),
                         arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))

    ax2.text(5, -0.5,
             "Features aprendidas automaticamente\ndurante o treinamento com dados rotulados",
             ha="center", fontsize=8, color="#555", style="italic")

    plt.tight_layout()
    saida = os.path.join("resultados", "05_pipeline_vs_cnn.png")
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"  Salvo em: {saida}")
    plt.show()


def imprimir_analise_cnn(objetos: list):
    n_p = sum(1 for o in objetos if o["classe"] == "parafuso")
    n_c = sum(1 for o in objetos if o["classe"] == "porca")
    n_i = sum(1 for o in objetos if o["classe"] == "indefinido")
    acertos = n_p + n_c   # esperado: 6 parafusos + 5 porcas

    print("=" * 62)
    print("ETAPA 5 — CNNs e Análise Crítica dos Resultados")
    print("=" * 62)

    print(f"""
RESULTADOS OBTIDOS:
  Parafusos classificados : {n_p}  (esperado: 6)
  Porcas classificadas    : {n_c}  (esperado: 5)
  Indefinidos             : {n_i}
  Total detectado         : {len(objetos)}

O QUE FUNCIONOU BEM:
  → O pipeline de pré-processamento (CLAHE + Otsu + morfologia)
    separou eficazmente os objetos metálicos do fundo claro.
  → O aspect ratio provou ser um discriminador eficiente para
    parafusos, cuja forma alongada gera valores > 1.8.
  → A circularidade e a solidez complementaram a classificação
    das porcas, que têm forma hexagonal compacta.
  → O pipeline é totalmente interpretável — cada decisão tem
    uma justificativa numérica rastreável.

LIMITAÇÕES OBSERVADAS:
  → O sistema depende de um fundo controlado (claro e uniforme).
    Em bancadas reais com sujeira ou iluminação variável, a
    limiarização de Otsu pode falhar.
  → Objetos sobrepostos ou muito próximos podem ser detectados
    como um único contorno, gerando classificação errada.
  → As regras foram calibradas manualmente — qualquer novo tipo
    de peça exigiria revisão dos limiares.
  → Sem invariância a escala: um parafuso muito pequeno pode ter
    aspect ratio similar a uma porca.

POSSÍVEIS MELHORIAS:
  → Usar detecção de bordas (Canny) em vez de limiarização simples
    para cenários com fundo não uniforme.
  → Adicionar watershed ou GrabCut para separar objetos sobrepostos.
  → Incluir mais características (momentos de Hu, histograma de cor)
    para aumentar a robustez da classificação.
  → Substituir as regras por um modelo de ML supervisionado
    (SVM ou Random Forest) treinado com exemplos rotulados.

─────────────────────────────────────────────────────────────
RELAÇÃO COM REDES NEURAIS CONVOLUCIONAIS (CNNs)
─────────────────────────────────────────────────────────────

O pipeline desenvolvido neste projeto é uma versão manual e
interpretável do que uma CNN faz automaticamente:

  PIPELINE MANUAL               CNN EQUIVALENTE
  ─────────────────────         ─────────────────────────────
  Cinza + blur + CLAHE    →     Normalização da entrada
  Limiarização + contorno →     Mapas de ativação (feature maps)
  Área, aspect ratio...   →     Features aprendidas (filtros conv.)
  Regras if/else          →     Camadas densas + softmax

A principal diferença é que no pipeline manual as características
são definidas pelo engenheiro com base no conhecimento do domínio.
Em uma CNN, essas características são APRENDIDAS automaticamente
a partir de milhares de exemplos rotulados.

VANTAGENS DE UMA CNN NESTE CENÁRIO:
  → Generaliza para variações de iluminação, ângulo e escala.
  → Detecta padrões sutis que regras manuais não capturam
    (ex: diferença entre parafuso M3 e M5 pelo perfil da rosca).
  → Escala facilmente para novas categorias sem reescrever regras.

QUANDO MANTER O PIPELINE MANUAL:
  → Quando os dados de treinamento são escassos (< 1000 imagens).
  → Quando a interpretabilidade é crítica (auditorias industriais).
  → Quando o ambiente é controlado e as regras são estáveis.
  → Para prototipagem rápida antes de investir em Deep Learning.
    """)

    print("=" * 62)
    print("  Etapa 5 concluída.\n")


def executar(objetos: list):
    imprimir_analise_cnn(objetos)
    exibir_pipeline_vs_cnn()


if __name__ == "__main__":
    from secao_01_gerar_imagem     import gerar_imagem_bancada
    from secao_02_preprocessamento import executar as pre
    from secao_03_deteccao         import executar as det
    from secao_04_classificacao    import executar as clas
    img    = gerar_imagem_bancada()
    etapas = pre(img)
    objs   = det(etapas)
    objs   = clas(img, objs)
    executar(objs)