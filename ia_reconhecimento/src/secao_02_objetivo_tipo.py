# =============================================================================
# SEÇÃO 2 — Objetivo do Projeto e Tipo de Problema de IA
# Projeto: Reconhecimento Facial para Controle de Acesso
# =============================================================================

def exibir_objetivo():
    print("=" * 70)
    print("SEÇÃO 2 — OBJETIVO DO PROJETO")
    print("=" * 70)

    print("""
OBJETIVO GERAL:
    Desenvolver uma solução de Inteligência Artificial capaz de comparar
    uma imagem facial capturada em tempo real com uma base de referência
    de rostos cadastrados, retornando uma decisão binária:
    ACESSO AUTORIZADO ou ACESSO NEGADO.

OBJETIVOS ESPECÍFICOS:
    → Carregar e organizar uma base de imagens de referência (rostos
      cadastrados de funcionários autorizados).
    → Capturar ou receber uma imagem de teste (rosto do candidato
      ao acesso).
    → Aplicar um modelo de verificação facial para comparar as imagens.
    → Calcular a similaridade entre os rostos e aplicar um limiar
      de decisão.
    → Exibir o resultado visual e registrar o log da tentativa de acesso
      (data, hora, resultado, nível de confiança).

RESULTADO ESPERADO AO FINAL DO DESENVOLVIMENTO:
    Um script funcional que, ao receber uma imagem de entrada, percorre
    a base de referência, compara os embeddings faciais e retorna:
    - Nome do funcionário reconhecido (se autorizado)
    - Nível de similaridade/confiança
    - Decisão: AUTORIZADO ✅ ou NEGADO ❌
    - Registro em arquivo de log com data e hora
    """)

    print("=" * 70)
    print("  Seção 2 concluída.\n")


def exibir_tipo_problema_ia():
    print("=" * 70)
    print("SEÇÃO 3 — TIPO DE PROBLEMA DE IA")
    print("=" * 70)

    print("""
CLASSIFICAÇÃO DO PROBLEMA:
    Este projeto envolve primariamente VERIFICAÇÃO FACIAL (1:1) com
    elementos de IDENTIFICAÇÃO FACIAL (1:N), conforme explicado abaixo.

┌─────────────────────┬──────────────────────────────────────────────────┐
│ Abordagem           │ Descrição                                        │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Verificação (1:1)   │ "Esta face é a mesma que a cadastrada para       │
│                     │  João Silva?" → Compara duas imagens específicas │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Identificação (1:N) │ "Quem é esta pessoa?" → Compara a face de        │
│                     │  entrada com TODA a base de cadastros            │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Nossa solução       │ Identificação 1:N: a imagem de entrada é         │
│                     │  comparada com todos os cadastros para encontrar │
│                     │  o mais similar e decidir o acesso               │
└─────────────────────┴──────────────────────────────────────────────────┘

TIPO DE APRENDIZADO ENVOLVIDO:
    → Aprendizado por Transferência (Transfer Learning): utilizamos
      modelos pré-treinados (VGG-Face, Facenet) que já aprenderam
      características faciais em milhões de imagens. Não treinamos
      um modelo do zero.

    → Aprendizado Métrico: o modelo não classifica diretamente, mas
      aprende a medir a DISTÂNCIA entre representações faciais
      (embeddings). Se a distância for menor que o limiar definido,
      as faces são consideradas da mesma pessoa.

TÉCNICA CENTRAL — EMBEDDINGS FACIAIS:
    Cada rosto é transformado em um vetor numérico (embedding) de
    128 ou 512 dimensões. Rostos da mesma pessoa geram vetores
    próximos no espaço vetorial; rostos diferentes geram vetores
    distantes. A comparação é feita por distância cosseno ou euclidiana.

    Imagem → [Modelo de IA] → Vetor [0.23, -0.87, 0.41, ...] (embedding)
    Comparação: distância(embedding_teste, embedding_referencia) < limiar
    """)

    print("=" * 70)
    print("  Seção 3 concluída.\n")
