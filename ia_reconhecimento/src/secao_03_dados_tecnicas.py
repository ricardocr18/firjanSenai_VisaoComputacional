# =============================================================================
# SEÇÃO 3 — Dados Necessários e Técnicas e Modelos
# Projeto: Reconhecimento Facial para Controle de Acesso
# =============================================================================

import os

def exibir_dados_necessarios():
    print("=" * 70)
    print("SEÇÃO 4 — DADOS NECESSÁRIOS")
    print("=" * 70)

    print("""
TIPOS DE IMAGENS NECESSÁRIAS:

  A) BASE DE REFERÊNCIA (imagens/referencia/)
     → Uma pasta por funcionário autorizado.
     → Mínimo recomendado: 3 a 5 fotos por pessoa, com variações de:
       - Iluminação (ambiente claro, sombra lateral)
       - Ângulo leve (frente, leve perfil)
       - Expressão neutra (padrão para cadastro)
     → Formato: .jpg ou .png, resolução mínima de 224x224 pixels.

     Estrutura esperada:
     imagens/referencia/
     ├── funcionario_01/
     │   ├── foto1.jpg
     │   ├── foto2.jpg
     │   └── foto3.jpg
     ├── funcionario_02/
     │   └── foto1.jpg
     └── ...

  B) IMAGEM DE TESTE (imagens/teste/)
     → Imagem capturada no momento da tentativa de acesso.
     → Pode vir de câmera ao vivo (webcam) ou arquivo estático.
     → O rosto deve estar centralizado e com boa iluminação.

ORGANIZAÇÃO DOS DADOS:
    → Cada subpasta em referencia/ representa um funcionário.
    → O nome da pasta é usado como identificador (ex: "joao_silva").
    → As imagens de teste não são armazenadas permanentemente —
      apenas o log do resultado é salvo em resultados/.

CUIDADOS ÉTICOS OBRIGATÓRIOS:
    ┌──────────────────────────────────────────────────────────────────┐
    │ ⚠️  DADOS BIOMÉTRICOS SÃO DADOS SENSÍVEIS (LGPD, Art. 11)       │
    ├──────────────────────────────────────────────────────────────────┤
    │ → Consentimento explícito: cada funcionário deve autorizar       │
    │   formalmente o uso de sua imagem facial.                        │
    │ → Finalidade específica: as imagens só podem ser usadas para     │
    │   controle de acesso, não para outros fins.                      │
    │ → Armazenamento seguro: imagens criptografadas, acesso restrito. │
    │ → Direito à exclusão: o funcionário pode solicitar remoção       │
    │   de seus dados a qualquer momento.                              │
    │ → Transparência: os funcionários devem saber que o sistema       │
    │   de reconhecimento facial está em uso.                          │
    │ → NÃO usar imagens de terceiros sem autorização.                 │
    │ → Para testes: usar bases públicas (LFW, CelebA) ou imagens      │
    │   sintéticas/fictícias.                                          │
    └──────────────────────────────────────────────────────────────────┘
    """)

    # Verifica estrutura de pastas do projeto
    print("  [Verificação da estrutura de pastas]")
    pastas = ["imagens/referencia", "imagens/teste", "resultados"]
    for pasta in pastas:
        status = "✅ existe" if os.path.exists(pasta) else "❌ não encontrada"
        print(f"  → {pasta:<25} {status}")
    print()

    print("=" * 70)
    print("  Seção 4 concluída.\n")


def exibir_tecnicas_modelos():
    print("=" * 70)
    print("SEÇÃO 5 — TÉCNICAS E MODELOS")
    print("=" * 70)

    print("""
BIBLIOTECA PRINCIPAL: DeepFace
    O DeepFace é uma biblioteca Python que unifica múltiplos modelos
    de reconhecimento facial em uma interface simples. Ela abstrai
    o pipeline completo: detecção → alinhamento → embedding → comparação.

MODELOS DISPONÍVEIS NO DEEPFACE:
┌──────────────┬────────────────┬─────────────────────────────────────────┐
│ Modelo       │ Embedding Size │ Característica                          │
├──────────────┼────────────────┼─────────────────────────────────────────┤
│ VGG-Face     │ 4096           │ Modelo clássico, alta precisão          │
│ Facenet      │ 128            │ Leve e rápido, bom para tempo real      │
│ Facenet512   │ 512            │ Versão mais precisa do Facenet          │
│ ArcFace      │ 512            │ Estado da arte, melhor precisão geral   │
│ DeepFace     │ 4096           │ Modelo original do Facebook             │
└──────────────┴────────────────┴─────────────────────────────────────────┘
  → Modelo escolhido para este projeto: ArcFace (melhor equilíbrio
    entre precisão e velocidade para uso em controle de acesso)

DETECTORES FACIAIS DISPONÍVEIS:
    → opencv   : rápido, menos preciso em ângulos
    → retinaface: preciso, detecta faces pequenas e com ângulo
    → mtcnn    : robusto, bom para múltiplas faces
    → Escolha: retinaface (já instalado como dependência)

MÉTRICAS DE DISTÂNCIA:
    → cosine           : mede ângulo entre vetores (0 = idêntico)
    → euclidean        : distância no espaço vetorial
    → euclidean_l2     : euclidiana normalizada (mais estável)
    → Escolha: cosine (padrão recomendado pelo DeepFace)

PIPELINE COMPLETO DA SOLUÇÃO:
    1. Imagem de entrada
         ↓
    2. Detecção facial (RetinaFace) — encontra e recorta o rosto
         ↓
    3. Alinhamento — normaliza posição dos olhos/nariz
         ↓
    4. Extração de embedding (ArcFace) — gera vetor de 512 dimensões
         ↓
    5. Comparação com embeddings da base de referência (distância cosseno)
         ↓
    6. Decisão: distância < limiar → AUTORIZADO | caso contrário → NEGADO
         ↓
    7. Registro em log (data, hora, nome, confiança, decisão)

BIBLIOTECAS COMPLEMENTARES:
    → OpenCV    : leitura/exibição de imagens e vídeo
    → NumPy     : operações matriciais nos embeddings
    → Matplotlib: visualização dos resultados
    """)

    print("=" * 70)
    print("  Seção 5 concluída.\n")
