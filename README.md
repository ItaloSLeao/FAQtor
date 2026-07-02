# FAQtor

FAQtor é um chatbot acadêmico de FAQ baseado em recuperação de informação. Ele recebe uma pergunta em linguagem natural e recupera a resposta mais relevante de uma base de FAQs universitárias, usando técnicas clássicas de Processamento de Linguagem Natural.

O projeto não usa LLMs, APIs externas ou geração de texto. A resposta é sempre recuperada da base cadastrada.

## Objetivo acadêmico

O objetivo é demonstrar um pipeline completo e interpretável de PLN para perguntas frequentes:

1. pré-processamento textual;
2. vetorização com TF-IDF;
3. cálculo de similaridade cosseno;
4. ranking top-k de respostas;
5. decisão por limiar de confiança;
6. sugestões simples por distância de edição;
7. avaliação experimental com métricas de ranking.

## Técnicas utilizadas

- normalização para minúsculas;
- remoção de acentos;
- tratamento de pontuação;
- tokenização simples;
- remoção opcional de stopwords;
- TF-IDF com unigramas e bigramas;
- similaridade de cosseno;
- ranking top-k;
- fallback por limiar de confiança;
- distância de Levenshtein;
- accuracy@1, accuracy@3 e Mean Reciprocal Rank.

## Base de dados

A base `data/faq.csv` contém 156 perguntas e respostas de atendimento acadêmico/universitário, cobrindo temas como matrícula, disciplinas, documentos, avaliações, TCC, estágio, biblioteca, bolsas, laboratórios e outros serviços acadêmicos.

A base `data/test_queries.csv` contém 152 consultas de teste, escritas como variações naturais das perguntas originais, incluindo linguagem informal, abreviações e pequenos erros de digitação.

As categorias incluem matrícula, trancamento, histórico, disciplinas, aproveitamento, TCC, estágio, biblioteca, monitoria, calendário, sistema acadêmico, avaliações, documentos, ementas, pré-requisitos, segunda chamada, prova final, frequência, jubilamento, transferência, reingresso, mobilidade acadêmica, bolsas e auxílios, restaurante universitário, transporte, laboratórios, projetos de pesquisa, extensão, iniciação científica, atendimento docente, coordenação, secretaria acadêmica, ENADE, formatura, carteira estudantil, declarações, cancelamento de matrícula e dados cadastrais.

## Arquitetura do pipeline

```text
Pergunta do usuário
        |
        v
Pré-processamento
        |
        v
TF-IDF da query
        |
        v
Similaridade cosseno contra perguntas da FAQ
        |
        v
Ranking top-k
        |
        v
Resposta recuperada ou fallback por limiar
        |
        v
Sugestões por Levenshtein quando a confiança é baixa
```

## Estrutura do projeto

```text
faqtor/
├── data/
│   ├── faq.csv
│   └── test_queries.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── vectorizer.py
│   ├── retriever.py
│   ├── spelling.py
│   ├── evaluation.py
│   └── config.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_spelling.py
│   └── test_retriever.py
├── app.py
├── main.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Instalação

Requisitos:

- Python 3.11 ou superior;
- pip.

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Em alguns sistemas, o comando pode ser `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Interface Streamlit

Execute:

```bash
streamlit run app.py
```

A interface permite digitar uma pergunta, ajustar o top-k e o limiar de confiança, além de visualizar a resposta principal, a pergunta mais similar, o score, a categoria, alternativas ranqueadas e sugestões por Levenshtein.

## CLI

Buscar uma resposta:

```bash
python main.py search "como faço matrícula?"
```

Executar a avaliação experimental:

```bash
python main.py evaluate
```

Também é possível alterar o top-k e o limiar:

```bash
python main.py search "como trancar uma disciplina?" --top-k 5 --threshold 0.25
```

## Exemplo de saída

```text
FAQtor
Pergunta: como faço matrícula?
Resposta: A matrícula semestral é feita pelo sistema acadêmico dentro do período definido no calendário.

Pergunta mais similar da base:
[1] Como faço minha matrícula no semestre?
Categoria: Matrícula
Score: 0.752
Confiante: sim
```

## Avaliação

O módulo `src/evaluation.py` executa a busca para cada consulta de teste e verifica se o `expected_id` aparece no top-1 e no top-3.

Exemplo de relatório:

```text
FAQtor - Relatório de avaliação
Consultas avaliadas: 152
Accuracy@1: 0.645
Accuracy@3: 0.803
Mean Reciprocal Rank: 0.718
```

Essas métricas podem variar se a base de FAQs, as consultas de teste ou os parâmetros forem alterados.

## Testes

Execute:

```bash
pytest
```

Os testes cobrem:

- normalização textual;
- tokenização;
- distância de Levenshtein;
- retorno de resultados na busca;
- fallback para consulta distante.

## Limitações

- O sistema depende de correspondência lexical; sinônimos muito diferentes podem reduzir a similaridade.
- A base de FAQs é sintética e pequena, criada para fins acadêmicos.
- Não há entendimento semântico profundo.
- A correção por Levenshtein sugere termos, mas não reescreve automaticamente a pergunta.
- O limiar de confiança precisa ser calibrado conforme a base.

## Melhorias futuras

- comparar Bag-of-Words, TF-IDF e LSA;
- aplicar LSA/SVD para reduzir dimensionalidade;
- adicionar classificação de intenção;
- usar uma base real de perguntas institucionais;
- criar uma interface web mais completa;
- analisar erros da avaliação por categoria;
- calibrar limiar com validação experimental.

## Licença

Este projeto segue a licença definida no arquivo `LICENSE`.
