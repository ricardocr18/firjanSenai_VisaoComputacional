# =============================================================================
# SEÇÃO 1 — Definição do Problema
# Projeto: Reconhecimento Facial para Controle de Acesso
# =============================================================================

def exibir_definicao_problema():
    print("=" * 70)
    print("SEÇÃO 1 — DEFINIÇÃO DO PROBLEMA")
    print("=" * 70)

    print("""
CONTEXTO:
    Uma empresa precisa controlar o acesso a uma área restrita de forma
    segura, eficiente e sem depender de crachás físicos ou senhas, que
    podem ser perdidos, esquecidos ou compartilhados indevidamente.

SITUAÇÃO-PROBLEMA:
    O sistema atual de controle de acesso é baseado em crachás físicos.
    Esse modelo apresenta os seguintes problemas:

    → Crachás podem ser emprestados ou clonados, comprometendo a segurança.
    → Funcionários frequentemente esquecem o crachá, gerando filas e
      atrasos na entrada.
    → Não há registro visual de quem realmente entrou — apenas de qual
      crachá foi usado.
    → O custo de reposição e manutenção de crachás é contínuo.

QUEM SERÁ BENEFICIADO:
    → Empresa: maior segurança e rastreabilidade dos acessos.
    → Setor de TI/Segurança: controle automatizado, com logs de entrada.
    → Funcionários: acesso mais ágil, sem depender de um objeto físico.

POR QUE O PROBLEMA É RELEVANTE:
    Em ambientes industriais, laboratórios e data centers, o controle
    de acesso inadequado pode gerar riscos à segurança física, roubo de
    propriedade intelectual e violação de normas regulatórias.
    A automação com reconhecimento facial elimina o fator humano de
    falha (esquecer o crachá) e aumenta a rastreabilidade.

PROPOSTA DE SOLUÇÃO:
    Desenvolver um sistema de IA que capture uma imagem do rosto do
    funcionário na entrada, compare com uma base de referência cadastrada
    e decida automaticamente se o acesso deve ser AUTORIZADO ou NEGADO,
    registrando data, hora e resultado de cada tentativa.
    """)

    print("=" * 70)
    print("  Seção 1 concluída.\n")
