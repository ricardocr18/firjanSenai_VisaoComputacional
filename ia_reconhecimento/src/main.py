# =============================================================================
# MAIN — Projeto de IA: Reconhecimento Facial para Controle de Acesso
# Executa todas as seções do projeto em sequência
# =============================================================================
# Autor  : Ricardo C. Ribeiro
# Curso  : Visão Computacional — FirjanSenai
# Prática: Estruturação do Projeto de IA — Reconhecimento Facial
# =============================================================================

import os
import sys
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Importa todos os módulos do projeto
sys.path.insert(0, os.path.dirname(__file__))
from secao_01_definicao_problema   import exibir_definicao_problema
from secao_02_objetivo_tipo        import exibir_objetivo, exibir_tipo_problema_ia
from secao_03_dados_tecnicas       import exibir_dados_necessarios, exibir_tecnicas_modelos
from secao_04_metricas_desafios    import exibir_metricas_avaliacao, exibir_desafios_limitacoes
from secao_05_reflexao_referencias import exibir_reflexao_final, exibir_referencias


# =============================================================================
# APRESENTAÇÃO DO PROJETO
# =============================================================================
def exibir_cabecalho():
    print()
    print("█" * 70)
    print("█                                                                    █")
    print("█     PROJETO DE IA — RECONHECIMENTO FACIAL PARA CONTROLE DE ACESSO █")
    print("█                                                                    █")
    print("█     Autor  : Ricardo C. Ribeiro                                   █")
    print("█     Curso  : Visão Computacional — FirjanSenai RJ                 █")
    print("█     Data   :", datetime.date.today().strftime("%d/%m/%Y"),
          "                                         █")
    print("█                                                                    █")
    print("█" * 70)
    print()


# =============================================================================
# DEMONSTRAÇÃO VISUAL DO PIPELINE
# Gera uma figura explicando o fluxo da solução com dados simulados
# =============================================================================
def gerar_visualizacao_pipeline():
    print("=" * 70)
    print("DEMONSTRAÇÃO VISUAL — Pipeline da Solução")
    print("=" * 70)

    os.makedirs("resultados", exist_ok=True)

    fig, eixos = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Pipeline: Reconhecimento Facial para Controle de Acesso",
        fontsize=14, fontweight="bold"
    )

    # --- Painel 1: Simulação de embedding facial ---
    ax1 = eixos[0]
    np.random.seed(10)
    # Simula dois embeddings (vetores de 20 dimensões para visualização)
    emb_referencia = np.random.normal(0.5, 0.15, 20)
    emb_autorizado = emb_referencia + np.random.normal(0, 0.05, 20)   # similar
    emb_negado     = np.random.normal(0.5, 0.15, 20) * -0.8           # diferente

    x = np.arange(20)
    ax1.plot(x, emb_referencia, 'b-o', markersize=4, label="Referência (cadastro)", linewidth=1.5)
    ax1.plot(x, emb_autorizado, 'g--s', markersize=4, label="Teste: AUTORIZADO", linewidth=1.5)
    ax1.plot(x, emb_negado,     'r:^', markersize=4, label="Teste: NEGADO", linewidth=1.5)
    ax1.set_title("Embeddings Faciais (20 dims — simulado)", fontsize=11)
    ax1.set_xlabel("Dimensão do vetor")
    ax1.set_ylabel("Valor do embedding")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Painel 2: Distribuição de distâncias cosseno ---
    ax2 = eixos[1]
    np.random.seed(42)
    dist_autorizado = np.random.normal(0.15, 0.05, 200)   # distâncias baixas = similar
    dist_negado     = np.random.normal(0.55, 0.10, 200)   # distâncias altas = diferente
    dist_autorizado = np.clip(dist_autorizado, 0, 1)
    dist_negado     = np.clip(dist_negado, 0, 1)

    ax2.hist(dist_autorizado, bins=25, alpha=0.6, color='green', label='Pessoas autorizadas')
    ax2.hist(dist_negado,     bins=25, alpha=0.6, color='red',   label='Pessoas não autorizadas')
    ax2.axvline(x=0.35, color='orange', linewidth=2.5, linestyle='--', label='Limiar de decisão (0.35)')
    ax2.set_title("Distribuição de Distâncias Cosseno", fontsize=11)
    ax2.set_xlabel("Distância cosseno")
    ax2.set_ylabel("Frequência")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Painel 3: Simulação de resultado de acesso ---
    ax3 = eixos[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title("Resultado do Sistema", fontsize=11)

    # Caso 1: Autorizado
    ax3.add_patch(plt.Rectangle((0.5, 5.5), 4, 3.5, color='#d4edda', ec='green', lw=2))
    ax3.text(2.5, 8.5, "✅  ACESSO AUTORIZADO", ha='center', va='center',
             fontsize=10, fontweight='bold', color='#155724')
    ax3.text(2.5, 7.7, "Funcionário: João Silva", ha='center', fontsize=9, color='#155724')
    ax3.text(2.5, 7.1, "Similaridade: 94.3%", ha='center', fontsize=9, color='#155724')
    ax3.text(2.5, 6.5, "Distância cosseno: 0.12", ha='center', fontsize=8, color='#155724')
    ax3.text(2.5, 5.9, "08/06/2025  08:47:23", ha='center', fontsize=8, color='gray')

    # Caso 2: Negado
    ax3.add_patch(plt.Rectangle((5.0, 5.5), 4, 3.5, color='#f8d7da', ec='red', lw=2))
    ax3.text(7.0, 8.5, "❌  ACESSO NEGADO", ha='center', va='center',
             fontsize=10, fontweight='bold', color='#721c24')
    ax3.text(7.0, 7.7, "Funcionário: Desconhecido", ha='center', fontsize=9, color='#721c24')
    ax3.text(7.0, 7.1, "Similaridade: 31.7%", ha='center', fontsize=9, color='#721c24')
    ax3.text(7.0, 6.5, "Distância cosseno: 0.68", ha='center', fontsize=8, color='#721c24')
    ax3.text(7.0, 5.9, "08/06/2025  09:13:05", ha='center', fontsize=8, color='gray')

    # Legenda do limiar
    ax3.add_patch(plt.Rectangle((0.5, 0.5), 9, 4.5, color='#fff3cd', ec='orange', lw=1.5))
    ax3.text(5.0, 4.6, "Limiar de decisão: distância cosseno < 0.35",
             ha='center', fontsize=9, fontweight='bold', color='#856404')
    ax3.text(5.0, 3.9, "Abaixo do limiar  → rostos similares → AUTORIZADO",
             ha='center', fontsize=8.5, color='green')
    ax3.text(5.0, 3.3, "Acima do limiar   → rostos diferentes → NEGADO",
             ha='center', fontsize=8.5, color='red')
    ax3.text(5.0, 2.5, "FAR alvo: < 0.1%   |   FRR alvo: < 5%",
             ha='center', fontsize=8.5, color='#856404')
    ax3.text(5.0, 1.7, "Modelo: ArcFace   |   Detector: RetinaFace",
             ha='center', fontsize=8.5, color='#856404')
    ax3.text(5.0, 1.0, "Métrica: Distância Cosseno",
             ha='center', fontsize=8.5, color='#856404')

    plt.tight_layout()

    caminho = os.path.join("resultados", "pipeline_reconhecimento_facial.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"\n  Visualização salva em: {caminho}")
    plt.show()
    print()


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    exibir_cabecalho()

    exibir_definicao_problema()
    exibir_objetivo()
    exibir_tipo_problema_ia()
    exibir_dados_necessarios()
    exibir_tecnicas_modelos()
    exibir_metricas_avaliacao()
    exibir_desafios_limitacoes()
    gerar_visualizacao_pipeline()
    exibir_reflexao_final()
    exibir_referencias()

    print("=" * 70)
    print("  PROJETO CONCLUÍDO — Todas as seções executadas com sucesso!")
    print(f"  Resultado salvo em: resultados/pipeline_reconhecimento_facial.png")
    print("=" * 70)
