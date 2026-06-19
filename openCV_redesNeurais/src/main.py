# =============================================================================
# main.py
# Desafio 01 SA — Detecção e Classificação de Objetos com OpenCV
# =============================================================================
# Script principal — executa todas as etapas do desafio em sequência:
#   01 → Geração da imagem sintética (bancada com parafusos e porcas)
#   02 → Pré-processamento (cinza, blur, CLAHE, Otsu, morfologia)
#   03 → Detecção de objetos e extração de características
#   04 → Classificação por regras visuais
#   05 → Análise crítica e relação com CNNs
# =============================================================================
# Autor  : Ricardo C. Ribeiro
# Curso  : Visão Computacional — FirjanSenai RJ
# Desafio: 01 SA — Detecção e Classificação com OpenCV e CNNs
# =============================================================================

import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(__file__))

from secao_01_gerar_imagem     import executar as etapa1
from secao_02_preprocessamento import executar as etapa2
from secao_03_deteccao         import executar as etapa3
from secao_04_classificacao    import executar as etapa4
from secao_05_cnn_analise      import executar as etapa5


def cabecalho():
    print()
    print("█" * 62)
    print("█                                                          █")
    print("█  DESAFIO 01 SA — DETECÇÃO E CLASSIFICAÇÃO COM OpenCV    █")
    print("█                                                          █")
    print("█  Autor  : Ricardo C. Ribeiro                            █")
    print("█  Curso  : Visão Computacional — FirjanSenai RJ          █")
    print("█  Data   :", datetime.date.today().strftime("%d/%m/%Y"),
          "                                     █")
    print("█                                                          █")
    print("█" * 62)
    print()


def referencias():
    print("=" * 62)
    print("REFERENCIAS")
    print("=" * 62)
    print("""
  [1] OpenCV — Open Source Computer Vision Library
      https://docs.opencv.org/

  [2] Bradski, G. (2000). The OpenCV Library.
      Dr. Dobb's Journal of Software Tools.

  [3] Otsu, N. (1979). A threshold selection method from
      gray-level histograms. IEEE Trans. Systems, Man, Cybernetics.

  [4] LeCun, Y. et al. (1998). Gradient-based learning applied
      to document recognition. Proceedings of the IEEE.

  [5] Goodfellow, I. et al. (2016). Deep Learning.
      MIT Press. http://www.deeplearningbook.org
    """)


if __name__ == "__main__":
    os.makedirs("imagens",   exist_ok=True)
    os.makedirs("resultados", exist_ok=True)

    cabecalho()

    # Etapa 1 — Gerar imagem
    img = etapa1()

    # Etapa 2 — Pré-processamento
    etapas = etapa2(img)

    # Etapa 3 — Detecção e características
    objetos = etapa3(etapas)

    # Etapa 4 — Classificação
    objetos = etapa4(img, objetos)

    # Etapa 5 — CNN e análise crítica
    etapa5(objetos)

    # Referências
    referencias()

    print("=" * 62)
    print("  DESAFIO CONCLUIDO — Resultados salvos em resultados/")
    print("  Arquivos gerados:")
    for f in sorted(os.listdir("resultados")):
        print(f"    → resultados/{f}")
    print("=" * 62)
