# =============================================================================
# 04_analise_etica.py
# Desafio 01 — Reconhecimento Facial para Controle de Acesso
# =============================================================================
# Analisa os resultados obtidos, discute limitações técnicas observadas
# e apresenta reflexão ética sobre o uso de reconhecimento facial.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def analisar_resultados(resultados: list):
    print("=" * 60)
    print("ETAPA 4 — Análise dos Resultados")
    print("=" * 60)

    print("""
RESUMO DOS CASOS TESTADOS:
""")
    acertos = 0
    for res in resultados:
        esperado   = "AUTORIZADO" if "autorizado" in res["label_teste"].upper() else "NEGADO"
        obtido     = "AUTORIZADO" if res["autorizado"] else "NEGADO"
        correto    = esperado == obtido
        if correto:
            acertos += 1
        marca = "OK" if correto else "ERRO"
        print(f"  [{marca}] {res['label_teste']}")
        print(f"       Esperado : {esperado}")
        print(f"       Obtido   : {obtido}")
        print(f"       Distância: {res['melhor_distancia']:.4f} | "
              f"Similaridade: {res['similaridade_pct']:.1f}%")
        print()

    acuracia = acertos / len(resultados) * 100
    print(f"  Acurácia nos casos testados: {acertos}/{len(resultados)} = {acuracia:.0f}%")

    print("""
O QUE FUNCIONOU BEM:
    → O DeepFace com ArcFace e RetinaFace identificou corretamente
      a correspondência entre fotos da mesma pessoa (mesma identidade,
      fotos diferentes do dataset LFW).
    → A distância cosseno se mostrou adequada para separar os casos:
      valor baixo para correspondência e alto para não-correspondência.
    → O pipeline completo (detecção → embedding → comparação → decisão)
      executou sem erros, demonstrando a viabilidade técnica da abordagem.

LIMITAÇÕES OBSERVADAS:
    → O dataset LFW contém imagens com variações de iluminação, ângulo
      e expressão. Isso pode aproximar as distâncias entre pessoas
      diferentes em condições similares.
    → Sem GPU, o processamento é mais lento — cada comparação pode
      levar alguns segundos, o que limitaria o uso em tempo real.
    → O limiar padrão (0.68 para ArcFace/cosseno) pode precisar de
      ajuste fino dependendo do nível de segurança exigido.
    → O sistema não possui detector de vivacidade (liveness detection),
      sendo vulnerável a ataques com fotos impressas.

POSSÍVEIS MELHORIAS:
    → Pré-calcular e armazenar os embeddings da base de referência
      para acelerar as comparações em produção.
    → Adicionar múltiplas fotos por pessoa na base de referência
      para aumentar a robustez da decisão.
    → Implementar liveness detection para evitar spoofing.
    → Calibrar o limiar de decisão com base em testes controlados
      com a população real de usuários.
    → Adicionar autenticação multifator (face + PIN) para áreas
      de alto risco.
    """)

    print("=" * 60)
    print("  Seção de análise concluída.\n")


def gerar_grafico_analise(resultados: list):
    """
    Gera gráfico comparativo das distâncias cosseno dos casos testados.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Análise dos Resultados — Distâncias e Decisões",
                 fontsize=13, fontweight="bold")

    # --- Gráfico 1: Distâncias por caso ---
    ax1 = eixos[0]
    labels    = [r["label_teste"].replace("Teste ", "").replace(" (", "\n(")
                 for r in resultados]
    distancias = [r["melhor_distancia"] for r in resultados]
    cores      = ["#2ecc71" if r["autorizado"] else "#e74c3c" for r in resultados]

    bars = ax1.bar(labels, distancias, color=cores, edgecolor="white",
                   linewidth=1.5, width=0.5)
    ax1.axhline(y=0.68, color="orange", linewidth=2.5, linestyle="--",
                label="Limiar de decisão (0.68)")
    ax1.set_title("Distância Cosseno por Caso de Teste", fontsize=11)
    ax1.set_ylabel("Distância cosseno")
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    for bar, dist in zip(bars, distancias):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 dist + 0.02, f"{dist:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    patch_auth = mpatches.Patch(color="#2ecc71", label="AUTORIZADO")
    patch_deny = mpatches.Patch(color="#e74c3c", label="NEGADO")
    ax1.legend(handles=[patch_auth, patch_deny,
               plt.Line2D([0], [0], color="orange", linewidth=2,
                          linestyle="--", label="Limiar (0.68)")],
               fontsize=9, loc="upper left")

    # --- Gráfico 2: Similaridade percentual ---
    ax2 = eixos[1]
    sims = [r["similaridade_pct"] for r in resultados]
    bars2 = ax2.bar(labels, sims, color=cores, edgecolor="white",
                    linewidth=1.5, width=0.5)
    ax2.axhline(y=32, color="orange", linewidth=2.5, linestyle="--",
                label="Limiar equiv. (32%)")
    ax2.set_title("Similaridade (%) por Caso de Teste", fontsize=11)
    ax2.set_ylabel("Similaridade (%)")
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, sim in zip(bars2, sims):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 sim + 1.5, f"{sim:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    caminho_saida = os.path.join("resultados", "04_analise_resultados.png")
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    print(f"  Gráfico de análise salvo em: {caminho_saida}")
    plt.show()


def reflexao_etica():
    print("=" * 60)
    print("ETAPA 5 — Reflexão Ética")
    print("=" * 60)

    print("""
O reconhecimento facial é uma tecnologia poderosa, mas seu uso envolve
responsabilidades técnicas, legais e humanas que não podem ser ignoradas.

1. PRIVACIDADE E CONSENTIMENTO
   → Dados faciais são dados biométricos — a categoria mais sensível
     de dados pessoais segundo a LGPD (Lei 13.709/2018, Art. 11).
   → Nenhum sistema de reconhecimento facial deve ser implantado sem
     consentimento explícito, informado e documentado de cada pessoa.
   → Os funcionários têm o direito de saber que estão sendo identificados,
     para qual finalidade e por quanto tempo os dados serão guardados.

2. RISCO DE ERRO E SUAS CONSEQUÊNCIAS
   → Falsos negativos bloqueiam pessoas autorizadas — geram transtorno
     e podem impedir acesso a áreas críticas em emergências.
   → Falsos positivos liberam acesso a pessoas não autorizadas — risco
     direto à segurança física e à propriedade da empresa.
   → Nenhum sistema de IA deve ser o único mecanismo de decisão em
     contextos de segurança. Revisão humana é essencial.

3. VIÉS ALGORÍTMICO
   → Modelos treinados com datasets não representativos podem ter
     desempenho inferior para certos grupos étnicos, gêneros ou
     faixas etárias — gerando discriminação involuntária.
   → O ArcFace foi treinado no MS-Celeb-1M, dataset com predominância
     de pessoas de determinadas origens. Isso deve ser considerado
     ao avaliar o sistema em populações diversas.

4. SEGURANÇA DOS DADOS
   → Imagens faciais armazenadas são alvos de ataques cibernéticos.
   → Se vazadas, não podem ser "trocadas" como uma senha — o rosto
     é permanente.
   → Recomendação: armazenar apenas os embeddings (vetores numéricos),
     não as imagens originais, e com criptografia.

5. RESPONSABILIDADE PROFISSIONAL
   → O profissional que desenvolve e implanta o sistema compartilha
     a responsabilidade pelo seu uso e pelos danos causados por erros.
   → É dever técnico documentar as limitações, comunicar os riscos
     e recomendar salvaguardas adequadas ao nível de risco da aplicação.
   → Este projeto é um exercício educacional e não deve ser tratado
     como sistema pronto para implantação real sem avaliação adicional.

CONCLUSÃO ÉTICA:
   A tecnologia de reconhecimento facial tem potencial real para
   melhorar a segurança e a eficiência em ambientes corporativos.
   Porém, seu uso responsável exige: consentimento, transparência,
   avaliação contínua de vieses, proteção de dados e supervisão humana.
   Ignorar esses aspectos transforma uma ferramenta útil em um risco
   para as pessoas que ela deveria proteger.
    """)

    print("=" * 60)
    print("  Reflexão ética concluída.\n")


if __name__ == "__main__":
    # Teste com dados simulados
    resultados_mock = [
        {
            "label_teste"      : "Teste autorizado (mesma pessoa)",
            "melhor_distancia" : 0.25,
            "similaridade_pct" : 75.0,
            "autorizado"       : True,
        },
        {
            "label_teste"      : "Teste negado (pessoa diferente)",
            "melhor_distancia" : 0.81,
            "similaridade_pct" : 19.0,
            "autorizado"       : False,
        },
    ]
    analisar_resultados(resultados_mock)
    gerar_grafico_analise(resultados_mock)
    reflexao_etica()
