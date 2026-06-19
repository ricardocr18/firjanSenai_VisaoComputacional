# =============================================================================
# secao_02_preprocessamento.py
# Desafio 01 SA — Detecção e Classificação de Objetos com OpenCV
# =============================================================================
# Aplica o pipeline de pré-processamento sobre a imagem da bancada:
#   1. Conversão para escala de cinza
#   2. Suavização com GaussianBlur
#   3. Ajuste de contraste (CLAHE)
#   4. Limiarização (Otsu) para separar objetos do fundo
#   5. Operações morfológicas para limpar ruídos
# =============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def aplicar_preprocessamento(img_bgr: np.ndarray) -> dict:
    """
    Aplica pipeline completo de pré-processamento.
    Retorna dicionário com todas as etapas intermediárias.
    """
    # 1. Escala de cinza
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Suavização — GaussianBlur reduz ruído sem destruir bordas
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)

    # 3. CLAHE — melhora contraste localmente (útil para peças metálicas
    #    que têm brilho similar ao fundo em algumas regiões)
    clahe     = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contraste = clahe.apply(suavizada)

    # 4. Limiarização de Otsu — calcula limiar ótimo automaticamente
    limiar_val, binaria = cv2.threshold(
        contraste, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5. Operações morfológicas
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Fechamento: fecha pequenos buracos dentro dos objetos
    fechada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Abertura: remove pequenos ruídos isolados do fundo
    limpa   = cv2.morphologyEx(fechada, cv2.MORPH_OPEN,  kernel, iterations=1)

    return {
        "original"  : img_bgr,
        "cinza"     : cinza,
        "suavizada" : suavizada,
        "contraste" : contraste,
        "binaria"   : binaria,
        "limpa"     : limpa,
        "limiar_val": limiar_val,
    }


def exibir_preprocessamento(etapas: dict):
    """Gera figura com todas as etapas do pré-processamento."""
    fig, eixos = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Pipeline de Pré-Processamento — Bancada Industrial",
        fontsize=14, fontweight="bold",
    )

    img_rgb = cv2.cvtColor(etapas["original"], cv2.COLOR_BGR2RGB)

    dados = [
        (eixos[0, 0], img_rgb,              "1. Original (RGB)",           None),
        (eixos[0, 1], etapas["cinza"],      "2. Escala de Cinza",          "gray"),
        (eixos[0, 2], etapas["suavizada"],  "3. GaussianBlur (5×5)",       "gray"),
        (eixos[1, 0], etapas["contraste"],  "4. CLAHE (contraste local)",  "gray"),
        (eixos[1, 1], etapas["binaria"],
         f"5. Limiarização Otsu (limiar={etapas['limiar_val']:.0f})",       "gray"),
        (eixos[1, 2], etapas["limpa"],
         "6. Morfologia (fechamento + abertura)",                            "gray"),
    ]

    for ax, img, titulo, cmap in dados:
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    saida = os.path.join("resultados", "02_preprocessamento.png")
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"  Salvo em: {saida}")
    plt.show()


def executar(img_bgr: np.ndarray) -> dict:
    print("=" * 62)
    print("ETAPA 2 — Pré-Processamento")
    print("=" * 62)

    etapas = aplicar_preprocessamento(img_bgr)

    print(f"""
ETAPAS APLICADAS:
  1. Escala de cinza    : reduz de 3 canais para 1 — simplifica análise
  2. GaussianBlur (5×5) : suaviza ruídos de textura do fundo
  3. CLAHE              : melhora contraste local das peças metálicas
  4. Limiarização Otsu  : limiar calculado = {etapas['limiar_val']:.0f}
                          separa objetos (escuros) do fundo (claro)
  5. Fechamento morfo.  : fecha buracos internos nos objetos
  6. Abertura morfo.    : remove ruídos isolados do fundo

POR QUE INVERTER NA LIMIARIZAÇÃO?
  As peças metálicas são mais escuras que o fundo de madeira clara.
  THRESH_BINARY_INV faz pixels escuros (peças) → brancos na máscara,
  permitindo que findContours detecte os objetos corretamente.
    """)

    exibir_preprocessamento(etapas)
    print("  Etapa 2 concluída.\n")
    return etapas


if __name__ == "__main__":
    from secao_01_gerar_imagem import gerar_imagem_bancada
    img = gerar_imagem_bancada()
    executar(img)
