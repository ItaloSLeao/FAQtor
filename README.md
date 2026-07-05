# FAQtor

FAQtor é um chatbot acadêmico baseado em recuperação de informação (TF-IDF e similaridade cosseno) que responde a perguntas frequentes usando uma base de dados própria, sem uso de LLMs ou APIs externas.

## Principais Características
- **Técnicas**: Pré-processamento textual, TF-IDF (uni/bigramas), similaridade cosseno, e correção por distância de Levenshtein.
- **Base de Dados**: 156 FAQs universitárias (`data/faq.csv`) e 152 consultas de teste (`data/test_queries.csv`).
- **Métricas**: Avaliado usando Accuracy@1, Accuracy@3 e Mean Reciprocal Rank (MRR).

## Instalação

Requer Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como Usar

**Interface Web (Streamlit):**
```bash
streamlit run app.py
```

**Linha de Comando (CLI):**
```bash
# Busca simples
python main.py search "como faço matrícula?"

# Busca com parâmetros
python main.py search "como trancar uma disciplina?" --top-k 5 --threshold 0.25

# Avaliar o modelo
python main.py evaluate

# Rodar testes automatizados
pytest
```

## Licença
Este projeto segue a licença definida no arquivo `LICENSE`.
