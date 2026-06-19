# =============================================================================
# secao_04_classificacao.py
# Desafio 01 SA — Detecção e Classificação de Objetos com OpenCV
# =============================================================================
# Classifica cada objeto detectado usando regras baseadas nas
# características visuais extraídas na etapa anterior.
#
# ANÁLISE DOS DADOS REAIS OBTIDOS NA ETAPA 3:
#
#   PORCAS (hexagonais):
#     → Circularidade alta: 0.82 a 0.85  (forma compacta e regular)
#     → Solidez alta      : 0.97 a 0.98  (casco convexo preenchido)
#     → Aspect ratio      : 0.95 a 1.12  (forma quadrada)
#
#   PARAFUSOS (alongados com cabeça circular):
#     → Circularidade baixa: 0.33 a 0.42 (forma irregular/alongada)
#     → Solidez média      : 0.76 a 0.81 (casco convexo menos preenchido)
#     → Aspect ratio variado (depende do ângulo na imagem)
#
# CRITÉRIO PRINCIPAL: circularidade + solidez
#   → Porca    : circularidade >= 0.70 E solidez >= 0.95
#   → Parafuso : circularidade <  0.55 E solidez <  0.90
#   → Indefinido: zona de transição
# =============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# LIMIARES CALIBRADOS COM BASE NOS DADOS REAIS DA ETAPA 3
# =============================================================================
LIMIAR_CIRC_PORCA    = 0.70   # porcas têm circularidade >= 0.70
LIMIAR_SOLID_PORCA   = 0.95   # porcas têm solidez >= 0.95
LIMIAR_CIRC_PARAF    = 0.55   # parafusos têm circularidade < 0.55
LIMIAR_SOLID_PARAF   = 0.90   # parafusos têm solidez < 0.90

COR_PARAFUSO   = (0,   200,  50)    # verde
COR_PORCA      = (50,  120, 255)    # azul
COR_INDEFINIDO = (0,   0,   200)    # vermelho


def classificar_objeto(obj: dict) -> str:
    """
    Classifica o objeto com base em circularidade e solidez.
    Esses dois critérios separam claramente porcas de parafusos
    nos dados reais obtidos neste cenário.
    """
    circ = obj["circularidade"]
    sol  = obj["solidez"]

    if circ >= LIMIAR_CIRC_PORCA and sol >= LIMIAR_SOLID_PORCA:
        return "porca"
    elif circ < LIMIAR_CIRC_PARAF and sol < LIMIAR_SOLID_PARAF:
        return "parafuso"
    else:
        return "indefinido"


def classificar_todos(objetos: list) -> list:
    for obj in objetos:
        obj["classe"] = classificar_objeto(obj)
    return objetos


def exibir_classificacao(img_bgr: np.ndarray, objetos: list):
    img_vis = img_bgr.copy()
    cores   = {
        "parafuso"  : COR_PARAFUSO,
        "porca"     : COR_PORCA,
        "indefinido": COR_INDEFINIDO,
    }

    for obj in objetos:
        x, y, larg, alt = obj["bbox"]
        cx, cy           = obj["cx"], obj["cy"]
        classe           = obj["classe"]
        cor              = cores[classe]

        cv2.rectangle(img_vis, (x, y), (x + larg, y + alt), cor, 3)
        cv2.drawContours(img_vis, [obj["contorno"]], -1, cor, 2)

        label = f"#{obj['id']} {classe}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(img_vis, (x, y - th - 10), (x + tw + 6, y), cor, -1)
        cv2.putText(img_vis, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.circle(img_vis, (cx, cy), 5, (255, 255, 255), -1)

    img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

    n_p = sum(1 for o in objetos if o["classe"] == "parafuso")
    n_c = sum(1 for o in objetos if o["classe"] == "porca")
    n_i = sum(1 for o in objetos if o["classe"] == "indefinido")

    fig, eixos = plt.subplots(1, 2, figsize=(16, 7),
                               gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Classificação de Objetos — Bancada Industrial",
                 fontsize=13, fontweight="bold")

    eixos[0].imshow(img_rgb)
    eixos[0].set_title("Resultado da Classificação por Regras Visuais\n"
                        "Verde = Parafuso | Azul = Porca | Vermelho = Indefinido",
                        fontsize=10)
    eixos[0].axis("off")

    patch_p = mpatches.Patch(color=(0/255, 200/255,  50/255), label=f"Parafuso ({n_p})")
    patch_c = mpatches.Patch(color=(50/255, 120/255, 255/255), label=f"Porca ({n_c})")
    patch_i = mpatches.Patch(color=(200/255, 0, 0),            label=f"Indefinido ({n_i})")
    eixos[0].legend(handles=[patch_p, patch_c, patch_i],
                    loc="lower right", fontsize=10,
                    facecolor="white", edgecolor="gray")

    ax2 = eixos[1]
    labels_pie, valores_pie, cores_pie = [], [], []
    mapa = {
        "Parafusos" : (n_p, (0/255,   200/255,  50/255)),
        "Porcas"    : (n_c, (50/255,  120/255, 255/255)),
        "Indefinido": (n_i, (200/255, 0,        0)),
    }
    for label, (val, cor) in mapa.items():
        if val > 0:
            labels_pie.append(f"{label}\n({val})")
            valores_pie.append(val)
            cores_pie.append(cor)

    ax2.pie(valores_pie, labels=labels_pie, colors=cores_pie,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 11})
    ax2.set_title(f"Total: {len(objetos)} objetos", fontsize=11)

    plt.tight_layout()
    saida = os.path.join("resultados", "04_classificacao.png")
    plt.savefig(saida, dpi=150, bbox_inches="tight")
    print(f"  Salvo em: {saida}")
    plt.show()


def imprimir_tabela_classificacao(objetos: list):
    print(f"\n  {'ID':>3} {'Classe':>12} {'Circular.':>10} "
          f"{'Solidez':>8} {'AspRatio':>9} {'Critério usado'}")
    print("  " + "-" * 72)

    for obj in objetos:
        circ = obj["circularidade"]
        sol  = obj["solidez"]
        ar   = obj["aspect_ratio"]
        cl   = obj["classe"]

        if cl == "porca":
            motivo = f"circ={circ:.3f}>={LIMIAR_CIRC_PORCA}, sol={sol:.3f}>={LIMIAR_SOLID_PORCA}"
        elif cl == "parafuso":
            motivo = f"circ={circ:.3f}<{LIMIAR_CIRC_PARAF}, sol={sol:.3f}<{LIMIAR_SOLID_PARAF}"
        else:
            motivo = f"zona ambígua: circ={circ:.3f}, sol={sol:.3f}"

        print(f"  {obj['id']:>3} {cl:>12} {circ:>10.3f} "
              f"{sol:>8.3f} {ar:>9.2f}  {motivo}")

    n_p = sum(1 for o in objetos if o["classe"] == "parafuso")
    n_c = sum(1 for o in objetos if o["classe"] == "porca")
    n_i = sum(1 for o in objetos if o["classe"] == "indefinido")

    print(f"""
  RESUMO:
    Parafusos detectados : {n_p}  (esperado: 6)
    Porcas detectadas    : {n_c}  (esperado: 5)
    Indefinidos          : {n_i}

  CRITÉRIOS APLICADOS:
    Porca    : circularidade >= {LIMIAR_CIRC_PORCA} E solidez >= {LIMIAR_SOLID_PORCA}
    Parafuso : circularidade <  {LIMIAR_CIRC_PARAF} E solidez <  {LIMIAR_SOLID_PARAF}
    Indefinido: zona de transição entre os critérios
    """)


def executar(img_bgr: np.ndarray, objetos: list) -> list:
    print("=" * 62)
    print("ETAPA 4 — Classificação por Regras Visuais")
    print("=" * 62)
    print(f"""
CRITÉRIO PRINCIPAL: circularidade + solidez
  Análise dos dados reais (etapa 3) mostrou que:
  → Porcas hexagonais têm circularidade 0.82–0.85 e solidez 0.97–0.98
    (forma compacta e regular — o casco convexo preenche quase todo o objeto)
  → Parafusos têm circularidade 0.33–0.42 e solidez 0.76–0.81
    (forma irregular — cabeça circular + haste alongada cria contorno complexo)

  O aspect ratio foi descartado como critério principal porque
  parafusos em ângulo diagonal aparecem com bounding box quadrado,
  tornando-o um discriminador pouco confiável neste cenário.
    """)

    objetos = classificar_todos(objetos)
    imprimir_tabela_classificacao(objetos)
    exibir_classificacao(img_bgr, objetos)
    print("  Etapa 4 concluída.\n")
    return objetos


if __name__ == "__main__":
    from secao_01_gerar_imagem     import gerar_imagem_bancada
    from secao_02_preprocessamento import executar as pre
    from secao_03_deteccao         import executar as det
    img    = gerar_imagem_bancada()
    etapas = pre(img)
    objs   = det(etapas)
    executar(img, objs)