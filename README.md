# FAQtor

FAQtor é um sistema acadêmico de perguntas frequentes que recupera somente
respostas cadastradas em `data/faq.csv`, sem LLMs ou APIs externas. A busca
principal usa embeddings semânticos; o TF-IDF original permanece como baseline
de comparação.

## Instalação após baixar o repositório

Requer Python 3.12 ou superior.

```bash
cd FAQtor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O `requirements.txt` usa o índice oficial do PyTorch para instalar a versão
CPU-only.

## Execução

```bash
# CLI interativa
python faq-cli.py

# Busca única pela linha de comando
python main.py search "como faço matrícula?"
python main.py search "como trancar uma disciplina?" --top-k 5 --threshold 0.54

# Avaliação comparativa entre TF-IDF e embeddings
python main.py evaluate

# Testes automatizados
pytest
```

## Primeiro carregamento

Na primeira busca, o `sentence-transformers` baixa o modelo
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Essa etapa exige
internet uma única vez; depois, o modelo é reutilizado pelo cache local do
Hugging Face sem tentar acessar a rede. A revisão avaliada dos pesos está fixada
na configuração para manter os resultados reproduzíveis.

As 156 perguntas são convertidas em embeddings densos normalizados e comparadas
à consulta por produto escalar, equivalente à similaridade de cosseno. O cache
persistente em `data/cache/` é validado por hash das perguntas e pelo nome do
modelo e sua revisão, sendo regenerado automaticamente quando necessário. A
resposta só é retornada quando a similaridade alcança o limiar configurado de
`0.54`. O relatório de avaliação inclui o sweep usado para calibrar esse valor e
expõe falsos positivos e falsos negativos; o limiar reduz, mas não elimina, erros.

O TF-IDF usa correspondência lexical com unigramas e bigramas e é executado
separadamente pelo comando de avaliação.

## Licença

Este projeto segue a licença definida no arquivo `LICENSE`.
