# =============================================================================
# secao_01_gerar_imagem.py
# Desafio 01 SA — Detecção e Classificação de Objetos com OpenCV
# =============================================================================
# CENÁRIO SIMULADO:
#   Uma bancada de triagem industrial contém parafusos e porcas misturados.
#   O sistema deve identificar cada peça, extrair características visuais
#   e classificá-las automaticamente para apoiar o processo de triagem.
#
# Esta seção gera a imagem sintética que simula a bancada de testes,
# com parafusos (retangulares/alongados) e porcas (hexagonais/circulares)
# posicionados sobre fundo texturizado.
# =============================================================================

import os
import cv2
import numpy as np


def gerar_imagem_bancada(seed: int = 42) -> np.ndarray:
    """
    Gera imagem sintética de bancada industrial com parafusos e porcas.
    Retorna a imagem BGR (formato OpenCV).
    """
    np.random.seed(seed)

    # --- Canvas: bancada de madeira clara texturizada ---
    h, w = 600, 900
    img = np.ones((h, w, 3), dtype=np.uint8)

    # Fundo: cor madeira clara com gradiente leve
    for i in range(h):
        intensidade = int(210 + (i / h) * 20)
        img[i, :] = [intensidade - 30, intensidade - 10, intensidade]

    # Textura de madeira: linhas horizontais sutis
    np.random.seed(seed)
    for _ in range(80):
        y  = np.random.randint(0, h)
        x1 = np.random.randint(0, w // 2)
        x2 = np.random.randint(w // 2, w)
        espessura = np.random.randint(1, 3)
        cor = tuple(int(c) - np.random.randint(5, 20) for c in img[y, x1].tolist())
        cor = tuple(max(0, c) for c in cor)
        cv2.line(img, (x1, y), (x2, y), cor, espessura)

    # ==========================================================================
    # OBJETOS — definidos como lista para reaproveitamento na detecção
    # Cada objeto tem: tipo, posição central (cx, cy) e parâmetros visuais
    # ==========================================================================

    objetos = []

    # --- PARAFUSOS (6 unidades) — formato alongado, cinza metálico ---
    parafusos = [
        {"cx": 120, "cy": 140, "ang": 30,  "comp": 90, "larg": 18},
        {"cx": 320, "cy": 80,  "ang": 0,   "comp": 110,"larg": 16},
        {"cx": 550, "cy": 200, "ang": 75,  "comp": 95, "larg": 18},
        {"cx": 720, "cy": 120, "ang": 45,  "comp": 100,"larg": 17},
        {"cx": 200, "cy": 420, "ang": 15,  "comp": 85, "larg": 16},
        {"cx": 650, "cy": 460, "ang": 60,  "comp": 105,"larg": 18},
    ]

    for p in parafusos:
        cx, cy = p["cx"], p["cy"]
        ang    = p["ang"]
        comp   = p["comp"]
        larg   = p["larg"]

        # Corpo do parafuso (retângulo rotacionado)
        rect   = ((cx, cy), (comp, larg), ang)
        box    = cv2.boxPoints(rect).astype(np.int32)
        cor_corpo = (160, 160, 165)
        cv2.fillPoly(img, [box], cor_corpo)
        cv2.polylines(img, [box], True, (100, 100, 105), 2)

        # Cabeça do parafuso (círculo na extremidade)
        rad_ang = np.radians(ang)
        hx = int(cx - (comp // 2) * np.cos(rad_ang))
        hy = int(cy - (comp // 2) * np.sin(rad_ang))
        cv2.circle(img, (hx, hy), larg, (140, 140, 145), -1)
        cv2.circle(img, (hx, hy), larg, (90, 90, 95), 2)

        # Rosca: linhas transversais ao longo do corpo
        for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
            rx = int(cx + (comp // 2 - comp * frac) * np.cos(rad_ang))
            ry = int(cy + (comp // 2 - comp * frac) * np.sin(rad_ang))
            perp_x = int(rx - (larg // 2) * np.sin(rad_ang))
            perp_y = int(ry + (larg // 2) * np.cos(rad_ang))
            perp_x2 = int(rx + (larg // 2) * np.sin(rad_ang))
            perp_y2 = int(ry - (larg // 2) * np.cos(rad_ang))
            cv2.line(img, (perp_x, perp_y), (perp_x2, perp_y2), (110, 110, 115), 1)

        objetos.append({"tipo": "parafuso", "cx": cx, "cy": cy})

    # --- PORCAS (5 unidades) — formato hexagonal, cinza mais escuro ---
    porcas = [
        {"cx": 420, "cy": 300, "raio": 32, "ang": 10},
        {"cx": 180, "cy": 280, "raio": 28, "ang": 25},
        {"cx": 600, "cy": 350, "raio": 30, "ang": 0},
        {"cx": 800, "cy": 350, "raio": 26, "ang": 15},
        {"cx": 380, "cy": 490, "raio": 31, "ang": 5},
    ]

    for pc in porcas:
        cx, cy = pc["cx"], pc["cy"]
        raio   = pc["raio"]
        ang    = pc["ang"]

        # Hexágono externo
        pontos = []
        for i in range(6):
            a = np.radians(60 * i + ang)
            px = int(cx + raio * np.cos(a))
            py = int(cy + raio * np.sin(a))
            pontos.append([px, py])
        pontos = np.array(pontos, np.int32)

        cor_porca = (110, 115, 120)
        cv2.fillPoly(img, [pontos], cor_porca)
        cv2.polylines(img, [pontos], True, (70, 75, 80), 2)

        # Furo central (círculo escuro)
        raio_furo = raio // 3
        cv2.circle(img, (cx, cy), raio_furo, (40, 40, 42), -1)
        cv2.circle(img, (cx, cy), raio_furo, (30, 30, 32), 2)

        # Chanfro: círculo interno leve
        cv2.circle(img, (cx, cy), int(raio * 0.7), (90, 95, 100), 1)

        objetos.append({"tipo": "porca", "cx": cx, "cy": cy})

    # Sombra leve sob cada objeto (realismo)
    for obj in objetos:
        cv2.circle(
            img,
            (obj["cx"] + 4, obj["cy"] + 4),
            8,
            (180, 175, 170),
            -1,
        )

    return img


def salvar_imagem(img: np.ndarray, nome: str = "bancada_industrial.jpg") -> str:
    os.makedirs("imagens", exist_ok=True)
    caminho = os.path.join("imagens", nome)
    cv2.imwrite(caminho, img)
    return caminho


def exibir_imagem_original(img: np.ndarray, caminho: str):
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        "Bancada Industrial — Parafusos e Porcas (Imagem Original)\n"
        "Cenário: triagem visual automatizada de peças metálicas",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    saida = os.path.join("resultados", "01_imagem_original.png")
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"  Salvo em: {saida}")
    plt.show()


def executar():
    print("=" * 62)
    print("ETAPA 1 — Cenário e Geração da Imagem Sintética")
    print("=" * 62)
    print("""
CENÁRIO SIMULADO:
  Uma bancada de triagem industrial contém peças metálicas
  misturadas: parafusos (alongados) e porcas (hexagonais).
  O sistema deve detectar cada peça, extrair características
  visuais e classificá-las automaticamente.

FINALIDADE DA ANÁLISE:
  Prototipar um pipeline de visão computacional que possa
  apoiar a triagem automatizada, reduzindo a necessidade de
  inspeção manual e aumentando a velocidade do processo.

IMAGEM GERADA:
  → 6 parafusos em ângulos variados
  → 5 porcas hexagonais em posições distribuídas
  → Fundo texturizado simulando bancada de madeira clara
  → Resolução: 900 x 600 pixels
    """)

    img     = gerar_imagem_bancada()
    caminho = salvar_imagem(img)
    h, w, c = img.shape
    print(f"  Imagem gerada: {caminho}  ({w}x{h}px | {c} canais)")
    exibir_imagem_original(img, caminho)
    print("  Etapa 1 concluída.\n")
    return img


if __name__ == "__main__":
    executar()
