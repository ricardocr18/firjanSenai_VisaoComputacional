# Prática 01 — Introdução à Visão Computacional com Python e OpenCV

## Estrutura do Projeto

```
pratica01_OpenCV/
│
├── pyproject.toml          # Configuração do projeto e dependências (Poetry)
├── poetry.lock             # Lock file gerado pelo Poetry
├── README.md               # Este arquivo
│
├── imagem/                 # Coloque aqui a imagem de teste
│   └── componente.jpg      # (ou .png — renomeie conforme sua imagem)
│
├── resultados/             # Imagens geradas pelo processamento
│
└── src/
    └── processamento.py    # Script principal da prática
```

## Pré-requisitos

- Python 3.10+
- Poetry instalado ([instruções](https://python-poetry.org/docs/))

## Como executar

### 1. Instalar dependências
```bash
poetry install
```

### 2. Ativar o ambiente virtual
```bash
poetry shell
```

### 3. Executar o script
```bash
python src/processamento.py
```

## Dependências utilizadas

| Biblioteca    | Finalidade                                      |
|---------------|-------------------------------------------------|
| opencv-python | Leitura, conversão e processamento de imagens   |
| numpy         | Manipulação matricial dos dados da imagem       |
| matplotlib    | Visualização e exibição das imagens processadas |


## 🖼️ Imagem de Entrada

A imagem utilizada (`imagem/componente.jpg`) é uma simulação de uma placa de circuito impresso (PCB) contendo:

- **Fundo verde escuro** — representa o substrato da placa
- **Trilhas laranja/douradas** — grade de interconexões em linhas horizontais e verticais
- **Resistores** — pequenos componentes com faixas coloridas (RGB) distribuídos ao longo das trilhas
- **Circuitos integrados (CIs)** — componentes quadrados de cor vinho escuro com pinos visíveis nas bordas
- **Capacitores/pontos de teste** — círculos dourados com símbolo "+"

> **Dimensões:** 640 × 480 pixels | **Canais:** 3 (RGB) | **Tipo:** uint8 (0–255)

---

## 📊 Resultado do Processamento

O script gera automaticamente a figura `resultados/resultado_processamento.png` com 4 subplots:

| # | Etapa | O que mostra |
|---|---|---|
| 1 | **Imagem Original (RGB)** | A placa completa com cores preservadas |
| 2 | **Escala de Cinza** | Trilhas aparecem em cinza claro; resistores e CIs com intensidades distintas |
| 3 | **Suavização — GaussianBlur (5×5)** | Ruídos granulares reduzidos; transições entre regiões mais suaves |
| 4 | **Limiarização — Simples (esq.) vs OTSU (dir.)** | Simples (limiar=127): só os capacitores circulares ficam brancos. OTSU (limiar=58): trilhas e todos os componentes segmentados com muito mais precisão |

> O OTSU calculou limiar = **58**, bem abaixo do fixo (127), pois o histograma é dominado por pixels escuros do fundo da PCB. Isso evidencia a vantagem do método adaptativo para imagens com iluminação não uniforme.

---


## 📥 Acesso (Download Facilitado)

1. 🚀 **Visualização Rápida:** [Abrir no Editor Web](https://github.dev/ricardocr18/firjanSenai_VisaoComputacional/tree/main/pratica01_OpenCV) (ou pressione `.` no teclado).
2. 📦 **Download Direto (.zip):** [Clique aqui para baixar apenas esta pasta](https://download-directory.github.io/?url=https://github.com/ricardocr18/firjanSenai_VisaoComputacional/tree/main/pratica01_OpenCV).