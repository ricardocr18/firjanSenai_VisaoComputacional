# =============================================================================
# main.py
# Desafio 01 — Reconhecimento Facial para Controle de Acesso
# =============================================================================
# Script principal — executa todas as etapas do desafio em sequência:
#   00 → Geração das imagens sintéticas
#   01 → Carregamento e exibição das imagens
#   02 → Comparação facial com DeepFace (ArcFace + opencv)
#   03 → Decisão de acesso e feedback visual com dados reais
#   04 → Análise dos resultados e reflexão ética com dados reais
# =============================================================================
# Autor  : Ricardo C. Ribeiro
# Curso  : Visão Computacional — FirjanSenai RJ
# Desafio: 01 — Reconhecimento Facial para Controle de Acesso
# =============================================================================

import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(__file__))

from secao_00_baixar_imagens import baixar_imagens
from secao_01_carregar       import carregar_e_exibir_imagens
from secao_02_comparacao     import executar_comparacoes
from secao_03_decisao        import gerar_feedback_visual, registrar_log
from secao_04_analise        import analisar_resultados, gerar_grafico_analise, reflexao_etica


def cabecalho():
    print()
    print("█" * 65)
    print("█                                                             █")
    print("█   DESAFIO 01 — RECONHECIMENTO FACIAL — CONTROLE DE ACESSO  █")
    print("█                                                             █")
    print("█   Autor  : Ricardo C. Ribeiro                              █")
    print("█   Curso  : Visão Computacional — FirjanSenai RJ            █")
    print("█   Data   :", datetime.date.today().strftime("%d/%m/%Y"),
          "                                        █")
    print("█                                                             █")
    print("█" * 65)
    print()


def referencias():
    print("=" * 65)
    print("REFERENCIAS E MATERIAIS CONSULTADOS")
    print("=" * 65)
    print("""
  [1] DeepFace — Lightweight Face Recognition Library
      https://github.com/serengil/deepface

  [2] ArcFace: Additive Angular Margin Loss for Deep Face Recognition
      Deng et al. (2019). CVPR. arXiv:1801.07698

  [3] LFW — Labeled Faces in the Wild (dataset público de referência)
      http://vis-www.cs.umass.edu/lfw/

  [4] RetinaFace: Single-Shot Multi-Level Face Localisation
      Deng et al. (2019). arXiv:1905.00641

  [5] LGPD — Lei Geral de Protecao de Dados Pessoais
      Lei n 13.709/2018 — https://www.planalto.gov.br

  [6] OpenCV — Open Source Computer Vision Library
      https://docs.opencv.org/
    """)


if __name__ == "__main__":
    cabecalho()

    # ------------------------------------------------------------------
    # ETAPA 0 — Geração das imagens sintéticas
    # ------------------------------------------------------------------
    print("[ETAPA 0] Gerando imagens sintéticas...")
    baixar_imagens()

    # ------------------------------------------------------------------
    # ETAPA 1 — Carregamento e exibição das imagens
    # ------------------------------------------------------------------
    print("[ETAPA 1] Carregando e exibindo imagens...")
    carregar_e_exibir_imagens()

    # ------------------------------------------------------------------
    # ETAPA 2 — Comparação facial com DeepFace (dados reais)
    # ------------------------------------------------------------------
    print("\n[ETAPA 2] Executando comparação facial com DeepFace...")
    print("          Aguarde — pode levar alguns minutos na primeira execucao.\n")
    resultados = executar_comparacoes()

    # ------------------------------------------------------------------
    # ETAPA 3 — Decisão e feedback visual com dados REAIS
    # ------------------------------------------------------------------
    print("\n[ETAPA 3] Gerando feedback visual com resultados reais...")
    gerar_feedback_visual(resultados)
    registrar_log(resultados)

    # ------------------------------------------------------------------
    # ETAPA 4 — Análise e reflexão ética com dados REAIS
    # ------------------------------------------------------------------
    print("\n[ETAPA 4] Analisando resultados e reflexão ética...")
    analisar_resultados(resultados)
    gerar_grafico_analise(resultados)
    reflexao_etica()

    # ------------------------------------------------------------------
    # Referências
    # ------------------------------------------------------------------
    referencias()

    print("=" * 65)
    print("  DESAFIO CONCLUIDO — Todos os resultados em resultados/")
    print("  Arquivos gerados:")
    for f in sorted(os.listdir("resultados")):
        print(f"    → resultados/{f}")
    print("=" * 65)