# =============================================================================
# 01_carregar_imagens.py
# Desafio 01 — Reconhecimento Facial para Controle de Acesso
# =============================================================================
# Carrega e exibe as imagens utilizadas no sistema, apresentando
# suas propriedades básicas (dimensões, canais, tipo de dado).
# =============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def carregar_e_exibir_imagens():
    print("=" * 60)
    print("ETAPA 1 — Carregamento e Exibição das Imagens")
    print("=" * 60)

    # --- Caminhos das imagens ---
    ref_dir  = os.path.join("imagens", "referencia")
    refs     = sorted([f for f in os.listdir(ref_dir) if f.endswith((".jpg", ".png"))])
    teste_autorizado = os.path.join("imagens", "teste", "teste_autorizado.jpg")
    teste_negado     = os.path.join("imagens", "teste", "teste_negado.jpg")

    if not refs:
        print("  [ERRO] Nenhuma imagem de referência encontrada.")
        print("  Execute primeiro: python src/00_baixar_imagens.py")
        return None, None, None

    # --- Carrega imagens de referência ---
    imagens_ref = []
    nomes_ref   = []
    print("\n  Imagens de referência carregadas:")
    for nome in refs:
        caminho = os.path.join(ref_dir, nome)
        img = cv2.imread(caminho)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagens_ref.append(img_rgb)
            nomes_ref.append(nome)
            h, w, c = img.shape
            print(f"    → {nome:30s} | {w}x{h}px | {c} canais | {img.dtype}")

    # --- Carrega imagens de teste ---
    img_auth = cv2.cvtColor(cv2.imread(teste_autorizado), cv2.COLOR_BGR2RGB)
    img_deny = cv2.cvtColor(cv2.imread(teste_negado),     cv2.COLOR_BGR2RGB)

    print("\n  Imagens de teste carregadas:")
    for label, img, path in [
        ("teste_autorizado.jpg", img_auth, teste_autorizado),
        ("teste_negado.jpg",     img_deny, teste_negado),
    ]:
        h, w, c = img.shape
        print(f"    → {label:30s} | {w}x{h}px | {c} canais | {img.dtype}")

    # --- Visualização ---
    n_ref = len(imagens_ref)
    fig, eixos = plt.subplots(1, n_ref + 2, figsize=(4 * (n_ref + 2), 5))
    fig.suptitle(
        "Imagens Utilizadas no Sistema de Controle de Acesso\n"
        "Fonte: LFW — Labeled Faces in the Wild (dataset público)",
        fontsize=12, fontweight="bold"
    )

    for i, (img, nome) in enumerate(zip(imagens_ref, nomes_ref)):
        eixos[i].imshow(img)
        eixos[i].set_title(f"Referência {i+1}\n{nome}", fontsize=9)
        eixos[i].axis("off")

    eixos[n_ref].imshow(img_auth)
    eixos[n_ref].set_title("Teste\n(mesma pessoa)", fontsize=9, color="green")
    eixos[n_ref].axis("off")

    eixos[n_ref + 1].imshow(img_deny)
    eixos[n_ref + 1].set_title("Teste\n(pessoa diferente)", fontsize=9, color="red")
    eixos[n_ref + 1].axis("off")

    plt.tight_layout()
    caminho_saida = os.path.join("resultados", "01_imagens_carregadas.png")
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    print(f"\n  Figura salva em: {caminho_saida}")
    plt.show()

    print("\n  [Interpretação]")
    print("  As imagens do LFW têm 250x250 pixels e 3 canais RGB.")
    print("  Cada pixel é representado por 3 valores inteiros (0-255),")
    print("  formando uma matriz de 250 x 250 x 3 = 187.500 valores.")
    print("  O sistema irá comparar os padrões faciais entre essas imagens.")
    print()

    return imagens_ref, img_auth, img_deny


if __name__ == "__main__":
    carregar_e_exibir_imagens()
