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


## 📥 Acesso (Download Facilitado)

1. 🚀 **Visualização Rápida:** [Abrir no Editor Web](https://github.dev/ricardocr18/firjanSenai_VisaoComputacional/tree/main/pratica01_pythonOpenCV) (ou pressione `.` no teclado).
2. 📦 **Download Direto (.zip):** [Clique aqui para baixar apenas esta pasta](https://download-directory.github.io/?url=https://github.com/ricardocr18/firjanSenai_VisaoComputacional/tree/main/pratica01_pythonOpenCV).