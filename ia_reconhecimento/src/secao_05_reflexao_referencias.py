# =============================================================================
# SEÇÃO 5 — Reflexão Final e Referências
# Projeto: Reconhecimento Facial para Controle de Acesso
# =============================================================================

def exibir_reflexao_final():
    print("=" * 70)
    print("SEÇÃO 8 — REFLEXÃO FINAL")
    print("=" * 70)

    print("""
O QUE APRENDI NA ESTRUTURAÇÃO DESTE PROJETO:

  Ao longo do planejamento desta solução de reconhecimento facial,
  ficou evidente que desenvolver um sistema de IA vai muito além de
  escolher uma biblioteca e escrever código. A etapa de estruturação
  revelou três aprendizados centrais:

  1. O PROBLEMA PRECISA SER BEM DEFINIDO ANTES DO CÓDIGO
     Antes de pensar em modelos ou técnicas, foi essencial entender
     quem seria beneficiado, quais são os riscos de falha e quais
     consequências cada tipo de erro (falso positivo vs falso negativo)
     teria no contexto real de segurança. Um falso positivo em controle
     de acesso é muito mais grave do que em um sistema de recomendação
     de filmes — isso muda completamente as escolhas técnicas.

  2. A ESCOLHA DO MODELO É UMA DECISÃO DE ENGENHARIA, NÃO DE SORTE
     Existem múltiplos modelos disponíveis no DeepFace (VGG-Face,
     Facenet, ArcFace). A escolha pelo ArcFace não foi aleatória —
     foi baseada na análise de precisão, tamanho do embedding e
     comportamento em cenários de controle de acesso documentados
     na literatura. Cada decisão técnica deve ter uma justificativa.

  3. ÉTICA NÃO É OPCIONAL — É PARTE DO PROJETO
     O reconhecimento facial envolve dados biométricos, que são
     considerados dados sensíveis pela LGPD. Ignorar consentimento,
     privacidade e viés algorítmico não é apenas um problema ético —
     é um risco legal e de reputação para a empresa. O planejamento
     ético deve acontecer junto com o planejamento técnico, não depois.

CUIDADOS PARA O DESENVOLVIMENTO DO DESAFIO FINAL:

  → Não usar imagens reais de pessoas sem consentimento explícito.
  → Usar bases públicas (ex: Labeled Faces in the Wild - LFW) para testes.
  → Pré-calcular embeddings da base de referência para melhor performance.
  → Definir e documentar o limiar de decisão com justificativa técnica.
  → Implementar log de acessos com data, hora e nível de confiança.
  → Testar o sistema com variações de iluminação e ângulo.
  → Avaliar FAR e FRR separadamente — acurácia geral pode ser enganosa.
  → Planejar o que acontece quando o sistema falha (plano de contingência).

CONCLUSÃO:
  Esta etapa de planejamento transformou um problema vago ("fazer
  reconhecimento facial") em uma proposta técnica estruturada, com
  decisões justificadas, riscos mapeados e responsabilidades definidas.
  O desenvolvimento do código será mais seguro, eficiente e responsável
  porque o planejamento foi feito com rigor.
    """)

    print("=" * 70)
    print("  Seção 8 concluída.\n")


def exibir_referencias():
    print("=" * 70)
    print("SEÇÃO 9 — REFERÊNCIAS E MATERIAIS CONSULTADOS")
    print("=" * 70)

    print("""
BIBLIOTECAS E DOCUMENTAÇÃO TÉCNICA:

  [1] DeepFace — Lightweight Face Recognition Library
      GitHub: https://github.com/serengil/deepface
      Serengil, S. I., & Ozpinar, A. (2020). LightFace: A Hybrid Deep
      Face Recognition Framework. INISTA 2020.

  [2] OpenCV — Open Source Computer Vision Library
      Documentação oficial: https://docs.opencv.org/

  [3] ArcFace: Additive Angular Margin Loss for Deep Face Recognition
      Deng et al. (2019). CVPR. arXiv:1801.07698

  [4] FaceNet: A Unified Embedding for Face Recognition and Clustering
      Schroff et al. (2015). CVPR. arXiv:1503.03832

LEGISLAÇÃO E ÉTICA:

  [5] Lei Geral de Proteção de Dados Pessoais (LGPD)
      Lei nº 13.709/2018 — Brasil
      https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm

  [6] ANPD — Autoridade Nacional de Proteção de Dados
      Guia de boas práticas para dados biométricos.
      https://www.gov.br/anpd

BASES DE DADOS PÚBLICAS PARA TESTES:

  [7] Labeled Faces in the Wild (LFW)
      Base pública para benchmark de reconhecimento facial.
      http://vis-www.cs.umass.edu/lfw/

  [8] CelebA Dataset — Large-scale Face Attributes Dataset
      http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

MATERIAL DO CURSO:

  [9] Roteiro de Estudos — Visão Computacional
      Cap. 1 (pág. 9-24): Fundamentos de Imagem Digital
      Cap. 2 (pág. 29-40): Redes Neurais Convolucionais
      Cap. 3 (pág. 45-52): Reconhecimento Facial e Aplicações
    """)

    print("=" * 70)
    print("  Seção 9 concluída.\n")
