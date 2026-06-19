# =============================================================================
# secao_00_baixar_imagens.py
# Desafio 01 — Reconhecimento Facial para Controle de Acesso
# =============================================================================
# Gera imagens sintéticas de rostos usando OpenCV e NumPy.
# Abordagem adotada após indisponibilidade dos servidores externos do LFW.
#
# JUSTIFICATIVA TÉCNICA:
#   Imagens sintéticas são amplamente usadas em pesquisas de visão
#   computacional para validar pipelines sem depender de dados reais.
#   Neste exercício educacional, simulamos dois perfis distintos:
#   → Perfil A (funcionário_01): rosto com tom de pele claro
#   → Perfil B (funcionario_02): rosto com tom de pele diferente
#   As variações entre as 3 fotos de referência simulam condições
#   reais de iluminação e expressão.
#
# NOTA ÉTICA:
#   Não são utilizadas imagens reais de pessoas. Todos os rostos
#   são completamente sintéticos e fictícios.
# =============================================================================

import os
import cv2
import numpy as np


def gerar_rosto(
    cor_pele,
    cor_cabelo,
    cor_olho,
    tamanho=(250, 250),
    ruido=8,
    brilho=0,
    seed=42,
):
    """
    Gera uma imagem sintética de rosto humano estilizado usando OpenCV.
    Parâmetros controlam cor de pele, cabelo, olhos, iluminação e ruído
    para simular variações entre fotos da mesma pessoa.
    """
    np.random.seed(seed)
    h, w = tamanho
    img  = np.ones((h, w, 3), dtype=np.uint8) * 200  # fundo cinza claro

    cx, cy = w // 2, h // 2

    # --- Rosto (elipse) ---
    cv2.ellipse(img, (cx, cy + 10), (70, 90), 0, 0, 360, cor_pele, -1)

    # --- Cabelo ---
    cv2.ellipse(img, (cx, cy - 50), (72, 55), 0, 180, 360, cor_cabelo, -1)
    cv2.rectangle(img, (cx - 72, cy - 90), (cx + 72, cy - 48), cor_cabelo, -1)

    # --- Sobrancelhas ---
    cv2.ellipse(img, (cx - 28, cy - 20), (18, 5), -10, 0, 360, cor_cabelo, -1)
    cv2.ellipse(img, (cx + 28, cy - 20), (18, 5),  10, 0, 360, cor_cabelo, -1)

    # --- Olhos ---
    cv2.ellipse(img, (cx - 28, cy),     (14, 9), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (cx + 28, cy),     (14, 9), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img,  (cx - 28, cy),      7,       cor_olho, -1)
    cv2.circle(img,  (cx + 28, cy),      7,       cor_olho, -1)
    cv2.circle(img,  (cx - 25, cy - 2),  2,       (0, 0, 0), -1)   # pupila
    cv2.circle(img,  (cx + 31, cy - 2),  2,       (0, 0, 0), -1)

    # --- Nariz ---
    pts_nariz = np.array([
        [cx, cy + 20], [cx - 10, cy + 40], [cx + 10, cy + 40]
    ], np.int32)
    cv2.polylines(img, [pts_nariz], True,
                  tuple(max(0, c - 30) for c in cor_pele), 2)

    # --- Boca ---
    cv2.ellipse(img, (cx, cy + 55), (22, 10), 0, 0, 180,
                (120, 60, 60), 2)

    # --- Orelhas ---
    cv2.ellipse(img, (cx - 70, cy + 10), (10, 18), 0, 90, 270, cor_pele, -1)
    cv2.ellipse(img, (cx + 70, cy + 10), (10, 18), 0, 270, 90, cor_pele, -1)

    # --- Pescoço ---
    cv2.rectangle(img, (cx - 20, cy + 95), (cx + 20, cy + 130), cor_pele, -1)

    # --- Ruído (simula textura de pele / variação de câmera) ---
    if ruido > 0:
        noise = np.random.randint(-ruido, ruido, img.shape, dtype=np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- Brilho (simula variação de iluminação) ---
    if brilho != 0:
        img = np.clip(img.astype(np.int16) + brilho, 0, 255).astype(np.uint8)

    return img


# =============================================================================
# Definição dos perfis sintéticos
# =============================================================================

# Perfil A — Funcionário 01 (3 fotos: variações de iluminação e ruído)
PERFIL_A = dict(cor_pele=(200, 160, 120), cor_cabelo=(40, 30, 20),
                cor_olho=(60, 40, 20))

# Perfil B — Funcionário 02 (1 foto de teste → deve ser NEGADO)
PERFIL_B = dict(cor_pele=(90, 130, 160), cor_cabelo=(20, 20, 60),
                cor_olho=(30, 80, 30))

IMAGENS = {
    os.path.join("imagens", "referencia", "funcionario_01_ref_a.jpg"):
        {**PERFIL_A, "ruido": 6,  "brilho":  0,  "seed": 1},
    os.path.join("imagens", "referencia", "funcionario_01_ref_b.jpg"):
        {**PERFIL_A, "ruido": 10, "brilho": -15, "seed": 2},
    os.path.join("imagens", "referencia", "funcionario_01_ref_c.jpg"):
        {**PERFIL_A, "ruido": 4,  "brilho":  20, "seed": 3},
    os.path.join("imagens", "teste", "teste_autorizado.jpg"):
        {**PERFIL_A, "ruido": 8,  "brilho":  10, "seed": 7},
    os.path.join("imagens", "teste", "teste_negado.jpg"):
        {**PERFIL_B, "ruido": 6,  "brilho":   0, "seed": 9},
}


def baixar_imagens():
    print("=" * 60)
    print("ETAPA 0 — Geração de Imagens Sintéticas")
    print("=" * 60)
    print("  Abordagem: geração local com OpenCV (sem dependência de rede)")
    print("  Justificativa: servidores LFW indisponíveis na rede atual.")
    print("  Todos os rostos são fictícios — nenhuma pessoa real.")
    print()

    os.makedirs(os.path.join("imagens", "referencia"), exist_ok=True)
    os.makedirs(os.path.join("imagens", "teste"),      exist_ok=True)
    os.makedirs("resultados",                           exist_ok=True)

    for caminho, params in IMAGENS.items():
        img = gerar_rosto(**params)
        cv2.imwrite(caminho, img)
        h, w = img.shape[:2]
        print(f"  Gerada → {caminho}  ({w}x{h}px)")

    print()
    print("  Imagens de referência:")
    for f in sorted(os.listdir(os.path.join("imagens", "referencia"))):
        print(f"    → {f}")

    print()
    print("  Imagens de teste:")
    for f in sorted(os.listdir(os.path.join("imagens", "teste"))):
        print(f"    → {f}")

    print()
    print("  Geração concluída com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    baixar_imagens()