# =============================================================================
# 03_decisao_feedback.py
# Desafio 01 — Reconhecimento Facial para Controle de Acesso
# =============================================================================
# Gera a decisão de acesso (AUTORIZADO / NEGADO) e o feedback visual
# para o usuário, exibindo a imagem de teste com o resultado destacado.
# =============================================================================

import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def gerar_feedback_visual(resultados: list):
    """
    Gera figura com feedback visual para cada caso de teste.
    Exibe a imagem de teste com moldura colorida e decisão textual.
    """
    print("=" * 60)
    print("ETAPA 3 — Classificação, Decisão e Feedback Visual")
    print("=" * 60)

    n = len(resultados)
    fig, eixos = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        eixos = [eixos]

    fig.suptitle(
        "Sistema de Controle de Acesso — Resultado da Verificação Facial",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, res in zip(eixos, resultados):
        # Carrega imagem de teste
        img = cv2.cvtColor(cv2.imread(res["caminho_teste"]), cv2.COLOR_BGR2RGB)

        autorizado   = res["autorizado"]
        cor_moldura  = "#2ecc71" if autorizado else "#e74c3c"   # verde / vermelho
        cor_texto    = "#155724" if autorizado else "#721c24"
        cor_fundo    = "#d4edda" if autorizado else "#f8d7da"
        icone        = "[AUTORIZADO]" if autorizado else "[NEGADO]"
        decisao      = "ACESSO AUTORIZADO" if autorizado else "ACESSO NEGADO"
        timestamp    = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

        # Exibe imagem
        ax.imshow(img)
        ax.set_title(res["label_teste"], fontsize=10, pad=8)
        ax.axis("off")

        # Moldura colorida ao redor da imagem
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(cor_moldura)
            spine.set_linewidth(6)

        # Caixa de resultado abaixo da imagem
        info = (
            f"{icone}  {decisao}\n"
            f"Similaridade : {res['similaridade_pct']:.1f}%\n"
            f"Distância    : {res['melhor_distancia']:.4f}\n"
            f"Referência   : {res['melhor_referencia']}\n"
            f"Data/Hora    : {timestamp}"
        )
        ax.text(
            0.5, -0.02, info,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5,
            color=cor_texto,
            bbox=dict(boxstyle="round,pad=0.6", facecolor=cor_fundo,
                      edgecolor=cor_moldura, linewidth=2),
        )

        # Imprime no terminal também
        print(f"\n  {decisao}")
        print(f"  Caso         : {res['label_teste']}")
        print(f"  Similaridade : {res['similaridade_pct']:.1f}%")
        print(f"  Distância    : {res['melhor_distancia']:.4f}")
        print(f"  Referência   : {res['melhor_referencia']}")
        print(f"  Data/Hora    : {timestamp}")

    plt.tight_layout()
    caminho_saida = os.path.join("resultados", "03_decisao_acesso.png")
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    print(f"\n  Figura salva em: {caminho_saida}")
    plt.show()
    print()


def registrar_log(resultados: list):
    """
    Registra todas as tentativas de acesso em arquivo de log .txt
    """
    os.makedirs("resultados", exist_ok=True)
    caminho_log = os.path.join("resultados", "log_acessos.txt")

    with open(caminho_log, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Sessão: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        for res in resultados:
            status = "AUTORIZADO" if res["autorizado"] else "NEGADO"
            f.write(
                f"[{status}] {res['label_teste']}\n"
                f"  Similaridade : {res['similaridade_pct']:.1f}%\n"
                f"  Distância    : {res['melhor_distancia']:.4f}\n"
                f"  Referência   : {res['melhor_referencia']}\n\n"
            )

    print(f"  Log registrado em: {caminho_log}")


if __name__ == "__main__":
    # Teste isolado com dados fictícios
    resultados_mock = [
        {
            "label_teste"      : "Teste AUTORIZADO (mesma pessoa)",
            "caminho_teste"    : os.path.join("imagens", "teste", "teste_autorizado.jpg"),
            "melhor_referencia": "bush_ref_01.jpg",
            "melhor_distancia" : 0.21,
            "similaridade_pct" : 79.0,
            "autorizado"       : True,
            "resultados"       : [],
        },
        {
            "label_teste"      : "Teste NEGADO (pessoa diferente)",
            "caminho_teste"    : os.path.join("imagens", "teste", "teste_negado.jpg"),
            "melhor_referencia": "bush_ref_01.jpg",
            "melhor_distancia" : 0.82,
            "similaridade_pct" : 18.0,
            "autorizado"       : False,
            "resultados"       : [],
        },
    ]
    gerar_feedback_visual(resultados_mock)
    registrar_log(resultados_mock)
