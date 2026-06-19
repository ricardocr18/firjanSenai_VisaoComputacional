# =============================================================================
# SEÇÃO 4 — Métricas de Avaliação e Desafios e Limitações
# Projeto: Reconhecimento Facial para Controle de Acesso
# =============================================================================

import numpy as np

def exibir_metricas_avaliacao():
    print("=" * 70)
    print("SEÇÃO 6 — MÉTRICAS DE AVALIAÇÃO")
    print("=" * 70)

    print("""
EM SISTEMAS DE CONTROLE DE ACESSO, OS ERROS TÊM CONSEQUÊNCIAS DIFERENTES:

┌──────────────────────┬───────────────────────────────────────────────────┐
│ Tipo de Resultado    │ Descrição                                         │
├──────────────────────┼───────────────────────────────────────────────────┤
│ Verdadeiro Positivo  │ Pessoa autorizada → sistema libera ✅             │
│ (TP)                 │ (acerto correto)                                  │
├──────────────────────┼───────────────────────────────────────────────────┤
│ Verdadeiro Negativo  │ Pessoa não autorizada → sistema bloqueia ✅       │
│ (TN)                 │ (acerto correto)                                  │
├──────────────────────┼───────────────────────────────────────────────────┤
│ Falso Positivo (FP)  │ Pessoa NÃO autorizada → sistema libera ❌         │
│ (Falsa Aceitação)    │ RISCO DE SEGURANÇA — mais crítico                 │
├──────────────────────┼───────────────────────────────────────────────────┤
│ Falso Negativo (FN)  │ Pessoa autorizada → sistema bloqueia ❌           │
│ (Falsa Rejeição)     │ Causa transtorno, mas não é risco de segurança    │
└──────────────────────┴───────────────────────────────────────────────────┘

MÉTRICAS PRINCIPAIS:

  1. FAR — False Acceptance Rate (Taxa de Falsa Aceitação)
     FAR = FP / (FP + TN)
     → Mede quantas vezes o sistema deixou entrar quem não devia.
     → Meta: FAR < 0.1% (crítico para segurança)

  2. FRR — False Rejection Rate (Taxa de Falsa Rejeição)
     FRR = FN / (FN + TP)
     → Mede quantas vezes o sistema bloqueou quem devia entrar.
     → Meta: FRR < 5% (aceitável para usabilidade)

  3. Acurácia Geral
     Acurácia = (TP + TN) / Total
     → Proporção de decisões corretas no total.

  4. Limiar de Decisão (Threshold)
     → O limiar de distância define o ponto de corte entre
       AUTORIZADO e NEGADO.
     → Limiar baixo: mais restritivo (menos FP, mais FN)
     → Limiar alto: mais permissivo (mais FP, menos FN)
     → Para segurança, preferimos limiares mais restritivos.

  5. Nível de Confiança (distância normalizada)
     → Exibido junto à decisão: ex. "Similaridade: 94.3%"
     → Permite auditoria dos casos limítrofes.
    """)

    # Simulação numérica das métricas com dados fictícios
    print("  [Simulação com dados fictícios para ilustração]")
    np.random.seed(42)
    n_testes = 100
    tp, tn, fp, fn = 72, 20, 3, 5

    acuracia = (tp + tn) / n_testes
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"  Total de testes simulados : {n_testes}")
    print(f"  Verdadeiros Positivos (TP): {tp}")
    print(f"  Verdadeiros Negativos (TN): {tn}")
    print(f"  Falsos Positivos (FP)     : {fp}  ← entrou quem não devia")
    print(f"  Falsos Negativos (FN)     : {fn}  ← bloqueou quem devia entrar")
    print(f"  Acurácia                  : {acuracia:.1%}")
    print(f"  FAR (Falsa Aceitação)     : {far:.1%}")
    print(f"  FRR (Falsa Rejeição)      : {frr:.1%}")
    print()

    print("=" * 70)
    print("  Seção 6 concluída.\n")


def exibir_desafios_limitacoes():
    print("=" * 70)
    print("SEÇÃO 7 — DESAFIOS E LIMITAÇÕES")
    print("=" * 70)

    print("""
DESAFIOS TÉCNICOS:

  1. QUALIDADE DA IMAGEM
     → Câmeras de baixa resolução geram embeddings menos precisos.
     → Compressão excessiva do JPEG apaga detalhes faciais sutis.
     → Solução: definir resolução mínima de captura (640x480).

  2. ILUMINAÇÃO
     → Luz lateral intensa cria sombras que alteram a percepção
       das distâncias entre pontos faciais.
     → Contraluz (janela atrás da pessoa) escurece o rosto.
     → Solução: iluminação frontal difusa no ponto de captura.

  3. ÂNGULO E POSE
     → Rostos em perfil ou com inclinação > 30° reduzem a precisão.
     → O alinhamento facial (etapa interna do DeepFace) minimiza
       esse problema, mas não elimina completamente.

  4. ACESSÓRIOS E MUDANÇAS FÍSICAS
     → Óculos, máscara, barba nova, mudança de corte de cabelo
       podem reduzir a similaridade com o cadastro.
     → Solução: recadastro periódico e múltiplas fotos de referência.

  5. VELOCIDADE DE PROCESSAMENTO
     → A extração de embedding pode ser lenta sem GPU.
     → Solução: pré-calcular e armazenar os embeddings da base
       de referência (não recalcular a cada acesso).

DESAFIOS ÉTICOS E DE PRIVACIDADE:

  6. VIÉS ALGORÍTMICO
     → Modelos treinados com datasets não representativos podem
       ter menor precisão para determinados grupos étnicos,
       faixas etárias ou gêneros.
     → Risco: sistema mais restritivo para alguns grupos,
       gerando discriminação involuntária.
     → Mitigação: avaliar o modelo com dados representativos
       da população de usuários reais.

  7. PRIVACIDADE E LGPD
     → Dados biométricos são dados sensíveis — exigem proteção
       reforçada pela Lei Geral de Proteção de Dados (LGPD).
     → O armazenamento não autorizado pode gerar penalidades legais.
     → Necessário: DPO (Encarregado de Dados), política de
       retenção e descarte de dados.

  8. RISCO DE USO INADEQUADO
     → O sistema não deve ser usado para monitoramento contínuo
       de funcionários além do controle de acesso.
     → Câmeras no local de trabalho exigem comunicação transparente.

  9. PONTO ÚNICO DE FALHA
     → Se o sistema cair, o acesso pode ser totalmente bloqueado.
     → Necessário: plano de contingência (crachá backup, segurança).

  10. DEEPFAKES E SPOOFING
     → Ataques com fotos impressas ou vídeos podem enganar o sistema.
     → Solução futura: detector de vivacidade (liveness detection).
    """)

    print("=" * 70)
    print("  Seção 7 concluída.\n")
