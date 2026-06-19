# =============================================================================
# Prática 01 — Introdução à Visão Computacional com Python e OpenCV
# =============================================================================
# Situação-Problema:
#   Uma empresa de inspeção visual de componentes industriais precisa
#   de um estudo inicial para automatizar a análise de imagens capturadas
#   em uma bancada de testes. Este script aplica técnicas básicas de
#   pré-processamento para melhorar a visualização e separar regiões
#   de interesse.
# =============================================================================

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)

# =============================================================================
# ETAPA 1 — Verificação do ambiente
# =============================================================================
print("=" * 60)
print("ETAPA 1 — Verificação do Ambiente")
print("=" * 60)
print(f"  Versão do OpenCV  : {cv2.__version__}")
print(f"  Versão do NumPy   : {np.__version__}")
print(f"  Versão do Python  : {sys.version.split()[0]}")
print()

# =============================================================================
# ETAPA 2 — Seleção e carregamento da imagem
# =============================================================================
# Caminho relativo à raiz do projeto.
# Coloque sua imagem na pasta "imagem/" e ajuste o nome abaixo se necessário.
CAMINHO_IMAGEM = os.path.join("imagem", "componente.jpg")

print("=" * 60)
print("ETAPA 2 / 3 — Carregando e Exibindo a Imagem")
print("=" * 60)

# O OpenCV lê imagens no padrão BGR (Blue-Green-Red).
# Para exibir corretamente com Matplotlib, que usa RGB, precisamos converter.
imagem_bgr = cv2.imread(CAMINHO_IMAGEM)

if imagem_bgr is None:
    print(f"\n[ERRO] Imagem não encontrada em: {CAMINHO_IMAGEM}")
    print("  → Verifique se a imagem foi colocada na pasta 'imagem/'")
    print("  → Verifique se o nome do arquivo corresponde ao da variável CAMINHO_IMAGEM")
    sys.exit(1)

# Convertendo de BGR (padrão OpenCV) para RGB (padrão Matplotlib)
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

print(f"  Imagem carregada com sucesso: {CAMINHO_IMAGEM}")
print()

# =============================================================================
# ETAPA 4 — Investigar propriedades da imagem
# =============================================================================
print("=" * 60)
print("ETAPA 4 — Propriedades da Imagem")
print("=" * 60)

altura, largura, canais = imagem_rgb.shape
tipo_dado = imagem_rgb.dtype
valor_minimo = imagem_rgb.min()
valor_maximo = imagem_rgb.max()

print(f"  Altura           : {altura} pixels")
print(f"  Largura          : {largura} pixels")
print(f"  Canais de cor    : {canais}  (R, G, B)")
print(f"  Tipo de dado     : {tipo_dado}  (uint8 = valores de 0 a 255)")
print(f"  Valor mínimo     : {valor_minimo}  (pixel mais escuro)")
print(f"  Valor máximo     : {valor_maximo}  (pixel mais claro)")
print()

# Análise interpretativa das propriedades
print("  [Interpretação]")
print(f"  A imagem é uma matriz de {altura} x {largura} posições.")
print(f"  Cada posição armazena 3 valores (R, G, B) entre 0 e 255.")
print(f"  Total de pixels: {altura * largura:,}")
print()

# =============================================================================
# ETAPA 5 — Converter para escala de cinza
# =============================================================================
print("=" * 60)
print("ETAPA 5 — Conversão para Escala de Cinza")
print("=" * 60)

# A conversão para cinza reduz a imagem de 3 canais para 1 canal.
# O OpenCV usa a fórmula: Y = 0.299·R + 0.587·G + 0.114·B
# Isso pondera cada canal de acordo com a sensibilidade do olho humano.
imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

print(f"  Dimensões após conversão: {imagem_cinza.shape}  (apenas 1 canal agora)")
print()
print("  [Por que converter para cinza?]")
print("  → Reduz a complexidade dos dados (3 canais → 1 canal).")
print("  → Algoritmos de limiarização e detecção de borda funcionam")
print("    sobre imagens de canal único.")
print("  → Elimina variações de cor irrelevantes para certas análises,")
print("    focando apenas na intensidade luminosa de cada pixel.")
print()

# =============================================================================
# ETAPA 6 — Aplicar suavização (GaussianBlur)
# =============================================================================
print("=" * 60)
print("ETAPA 6 — Suavização com GaussianBlur")
print("=" * 60)

# O GaussianBlur aplica uma média ponderada na vizinhança de cada pixel,
# dando mais peso ao centro do kernel. O resultado é uma imagem mais suave.
# kernel_size=(5, 5): área de influência de cada pixel ao redor
# sigmaX=0: o desvio padrão é calculado automaticamente pelo OpenCV
imagem_suavizada = cv2.GaussianBlur(imagem_cinza, ksize=(5, 5), sigmaX=0)

print("  Filtro aplicado : GaussianBlur  |  kernel: (5x5)  |  sigmaX: automático")
print()
print("  [O que esse filtro faz?]")
print("  → Reduz ruídos e pequenas variações de intensidade.")
print("  → Suaviza bordas excessivamente ruidosas.")
print("  → Prepara a imagem para a limiarização, evitando que")
print("    ruídos sejam interpretados como regiões de interesse.")
print()

# =============================================================================
# ETAPA 7 — Aplicar limiarização
# =============================================================================
print("=" * 60)
print("ETAPA 7 — Limiarização (Threshold)")
print("=" * 60)

# --- Limiarização simples ---
# Pixels com intensidade > 127 viram branco (255); os demais, preto (0).
limiar_simples = 127
_, imagem_thresh_simples = cv2.threshold(
    imagem_suavizada, limiar_simples, 255, cv2.THRESH_BINARY
)

# --- Limiarização automática pelo método OTSU ---
# O OTSU calcula automaticamente o melhor limiar para separar fundo e objeto,
# minimizando a variância dentro de cada classe (fundo e objeto).
limiar_otsu, imagem_thresh_otsu = cv2.threshold(
    imagem_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"  Limiarização simples : limiar fixo = {limiar_simples}")
print(f"  Limiarização OTSU    : limiar calculado automaticamente = {limiar_otsu:.0f}")
print()
print("  [Diferença entre os métodos]")
print("  → Simples: usa valor fixo escolhido manualmente. Sensível à")
print("    iluminação e pode não funcionar bem em condições variadas.")
print(f"  → OTSU: calculou {limiar_otsu:.0f} como ponto ótimo para esta imagem,")
print("    adaptando-se automaticamente ao histograma de intensidades.")
print()

# =============================================================================
# ETAPA 8 — Organizar visualizações com subplots
# =============================================================================
print("=" * 60)
print("ETAPA 8 — Gerando Visualização com Subplots")
print("=" * 60)

fig, eixos = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Prática 01 — Pré-processamento de Imagem com OpenCV",
    fontsize=15,
    fontweight="bold",
)

# Subplot 1: Imagem original colorida
eixos[0, 0].imshow(imagem_rgb)
eixos[0, 0].set_title("1. Imagem Original (RGB)", fontsize=12)
eixos[0, 0].axis("off")

# Subplot 2: Escala de cinza
eixos[0, 1].imshow(imagem_cinza, cmap="gray")
eixos[0, 1].set_title("2. Escala de Cinza", fontsize=12)
eixos[0, 1].axis("off")

# Subplot 3: Imagem suavizada
eixos[1, 0].imshow(imagem_suavizada, cmap="gray")
eixos[1, 0].set_title("3. Suavização — GaussianBlur (5×5)", fontsize=12)
eixos[1, 0].axis("off")

# Subplot 4: Comparação limiarização simples vs OTSU lado a lado
# Cria imagem composta para comparar os dois resultados num único subplot
comparacao = np.hstack([imagem_thresh_simples, imagem_thresh_otsu])
eixos[1, 1].imshow(comparacao, cmap="gray")
eixos[1, 1].set_title(
    f"4. Limiarização — Simples (limiar=127) | OTSU (limiar={limiar_otsu:.0f})",
    fontsize=10,
)
eixos[1, 1].axis("off")

plt.tight_layout()

# Salva a figura na pasta resultados/
caminho_saida = os.path.join("resultados", "resultado_processamento.png")
plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
print(f"  Figura salva em: {caminho_saida}")
print()

# Exibe a figura na tela
plt.show()

# =============================================================================
# ETAPA 9 — Interpretação dos resultados
# =============================================================================
print("=" * 60)
print("ETAPA 9 — Interpretação dos Resultados")
print("=" * 60)
print()
print("  [Regiões destacadas]")
print("  → A conversão para cinza permitiu visualizar variações de")
print("    luminosidade sem a influência das cores.")
print("  → A suavização reduziu ruídos presentes na imagem,")
print("    tornando as transições entre regiões mais uniformes.")
print("  → A limiarização separou regiões claras (branco) das")
print("    regiões escuras (preto), evidenciando contornos e áreas")
print("    de contraste.")
print()
print("  [Limitações observadas]")
print("  → Variações de iluminação na imagem podem fazer com que")
print("    regiões do mesmo objeto recebam classificações diferentes")
print("    na limiarização simples.")
print(f"  → O OTSU calculou limiar = {limiar_otsu:.0f}, que tende a ser mais")
print("    adequado, mas ainda pode perder detalhes em sombras.")
print("  → A suavização com kernel 5×5 pode apagar bordas finas,")
print("    o que seria problemático em análises de detalhes pequenos.")
print()
print("=" * 60)
print("ETAPA 10 — Script concluído com sucesso!")
print("=" * 60)
