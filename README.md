# Trabalho 3ª Etapa — Inteligência Artificial

Projeto acadêmico sobre RNNs, Transformers e modelos de linguagem, com um experimento local de geração de código usando Ollama.

## Conteúdo

- `Trabalho_Etapa_3_Relatorio.md`: pesquisa teórica e análise do experimento.
- `ollama_stream.py`: cliente Python que exibe a geração em streaming e salva as respostas.
- `experimento_temp_*.txt`: saídas observadas nas temperaturas 0.0, 0.9 e 1.5.

## Pré-requisitos

- Python 3.9 ou superior.
- Ollama instalado e em execução.
- Modelo `qwen2.5-coder:7b` baixado localmente.

No Linux, instale o Ollama pelo [instalador oficial](https://ollama.com/download/linux) e baixe o modelo:

```bash
ollama pull qwen2.5-coder:7b
```

Se o serviço não iniciar automaticamente, abra outro terminal e execute `ollama serve`.

Instale a dependência Python:

```bash
python -m pip install -r requirements.txt
```

## Uso

Uma geração individual, salva em `resposta_ollama.txt`:

```bash
python ollama_stream.py --temperature 0.2 --top-p 0.9 --num-predict 600
```

O experimento executa o mesmo prompt nas três temperaturas e salva um arquivo para cada resposta:

```bash
python ollama_stream.py --experimento --top-p 0.9 --num-predict 600
```

Para repetir exatamente o limite usado nos resultados registrados:

```bash
python ollama_stream.py --experimento --top-p 0.9 --num-predict 300
```

Os parâmetros `--model`, `--prompt`, `--url`, `--timeout` e `--output` podem ser ajustados. Em modo de experimento, os arquivos são salvos na pasta indicada por `--output` (na pasta atual por padrão). A geração por CPU pode ser lenta. Se o limite de tokens for atingido antes do fim da resposta, aumente `--num-predict`.

## Resultados

Os resultados registrados usam `qwen2.5-coder:7b`, `top_p=0.9` e `num_predict=300`. As três gerações terminaram no limite configurado; por isso os exemplos de código estão incompletos. O relatório descreve as diferenças observadas sem atribuir à temperatura um efeito determinístico de “caos”.
