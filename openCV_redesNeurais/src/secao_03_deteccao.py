# =============================================================================
# secao_03_deteccao.py
# Desafio 01 SA — Detecção e Classificação de Objetos com OpenCV
# =============================================================================
# Detecta objetos na máscara binária usando findContours.
# Para cada objeto detectado, extrai características visuais:
#   → Área (px²)
#   → Perímetro (px)
#   → Dimensões do bounding box (largura x altura)
#   → Razão largura/altura (aspect ratio)
#   → Circularidade (4π·área / perímetro²)
#   → Cor média na região do objeto (canal cinza)
#   → Solidez (área / área do casco convexo)
# =============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Área mínima para filtrar ruídos pequenos
AREA_MINIMA = 400


def extrair_contornos(mascara: np.ndarray) -> list:
    """Encontra e filtra contornos por área mínima."""
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contornos if cv2.contourArea(c) >= AREA_MINIMA]


def extrair_caracteristicas(contorno, img_cinza: np.ndarray) -> dict:
    """
    Extrai conjunto completo de características visuais de um contorno.
    """
    area      = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)

    # Bounding box
    x, y, larg, alt = cv2.boundingRect(contorno)

    # Aspect ratio: > 2.0 sugere objeto alongado (parafuso)
    aspect_ratio = larg / alt if alt > 0 else 0

    # Circularidade: 1.0 = círculo perfeito; < 0.5 = muito irregular
    circularidade = 0.0
    if perimetro > 0:
        circularidade = (4 * np.pi * area) / (perimetro ** 2)

    # Cor média na região do bounding box (canal cinza)
    roi       = img_cinza[y:y + alt, x:x + larg]
    cor_media = float(np.mean(roi)) if roi.size > 0 else 0.0

    # Solidez: área / área do casco convexo
    casco  = cv2.convexHull(contorno)
    area_casco = cv2.contourArea(casco)
    solidez = area / area_casco if area_casco > 0 else 0.0

    # Centro do objeto
    M  = cv2.moments(contorno)
    cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + larg // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + alt  // 2

    return {
        "contorno"     : contorno,
        "area"         : area,
        "perimetro"    : perimetro,
        "bbox"         : (x, y, larg, alt),
        "aspect_ratio" : aspect_ratio,
        "circularidade": circularidade,
        "cor_media"    : cor_media,
        "solidez"      : solidez,
        "cx"           : cx,
        "cy"           : cy,
    }


def detectar_objetos(etapas: dict) -> list:
    """
    Detecta todos os objetos e retorna lista de características.
    """
    contornos = extrair_contornos(etapas["limpa"])
    img_cinza = etapas["cinza"]

    objetos = []
    for i, c in enumerate(contornos):
        feat = extrair_caracteristicas(c, img_cinza)
        feat["id"] = i + 1
        objetos.append(feat)

    return objetos


def exibir_deteccao(img_bgr: np.ndarray, objetos: list):
    """Desenha contornos e bounding boxes sobre a imagem original."""
    img_vis = img_bgr.copy()

    for obj in objetos:
        x, y, larg, alt = obj["bbox"]
        cx, cy           = obj["cx"], obj["cy"]

        # Contorno branco
        cv2.drawContours(img_vis, [obj["contorno"]], -1, (255, 255, 255), 2)

        # Bounding box amarela
        cv2.rectangle(img_vis, (x, y), (x + larg, y + alt), (0, 220, 255), 2)

        # ID do objeto
        cv2.putText(
            img_vis, f"#{obj['id']}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2,
        )

        # Centro
        cv2.circle(img_vis, (cx, cy), 4, (0, 0, 255), -1)

    img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        f"Detecção de Objetos — {len(objetos)} objetos encontrados\n"
        "Contorno branco | Bounding box amarela | Centro vermelho",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    saida = os.path.join("resultados", "03_deteccao.png")
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"  Salvo em: {saida}")
    plt.show()


def imprimir_tabela(objetos: list):
    """Imprime tabela formatada com características de cada objeto."""
    print(f"\n  {'ID':>3} {'Área':>8} {'Perímetro':>10} "
          f"{'BBox (LxA)':>12} {'AspRatio':>9} "
          f"{'Circular.':>10} {'CorMédia':>9} {'Solidez':>8}")
    print("  " + "-" * 78)
    for obj in objetos:
        x, y, larg, alt = obj["bbox"]
        print(
            f"  {obj['id']:>3} "
            f"{obj['area']:>8.0f} "
            f"{obj['perimetro']:>10.1f} "
            f"{larg:>5}x{alt:<5} "
            f"{obj['aspect_ratio']:>9.2f} "
            f"{obj['circularidade']:>10.3f} "
            f"{obj['cor_media']:>9.1f} "
            f"{obj['solidez']:>8.3f}"
        )


def executar(etapas: dict) -> list:
    print("=" * 62)
    print("ETAPA 3 — Detecção de Objetos e Extração de Características")
    print("=" * 62)
    print(f"""
MÉTODO: cv2.findContours (RETR_EXTERNAL + CHAIN_APPROX_SIMPLE)
  → Detecta apenas contornos externos (ignora buracos internos)
  → Filtra objetos com área < {AREA_MINIMA}px² (elimina ruídos)

CARACTERÍSTICAS EXTRAÍDAS POR OBJETO:
  → Área          : tamanho em pixels quadrados
  → Perímetro     : comprimento do contorno em pixels
  → Bounding Box  : retângulo delimitador (x, y, largura, altura)
  → Aspect Ratio  : largura / altura — parafusos têm ratio alto
  → Circularidade : 4π·área/perímetro² — porcas têm valor próximo a 0.7
  → Cor Média     : intensidade média na região (cinza)
  → Solidez       : área / área do casco convexo
    """)

    objetos = detectar_objetos(etapas)
    print(f"  Objetos detectados: {len(objetos)}")
    imprimir_tabela(objetos)
    exibir_deteccao(etapas["original"], objetos)
    print("\n  Etapa 3 concluída.\n")
    return objetos


if __name__ == "__main__":
    from secao_01_gerar_imagem   import gerar_imagem_bancada
    from secao_02_preprocessamento import executar as pre
    img    = gerar_imagem_bancada()
    etapas = pre(img)
    executar(etapas)
