# Trabalho 3ª Etapa — Inteligência Artificial

**Centro Universitário Presidente Antônio Carlos — UNIPAC Barbacena**  
**Curso:** Ciência da Computação  
**Disciplina:** Inteligência Artificial  
**Professor:** Felipe Roncalli de Paula Carneiro  
**Data indicada no enunciado:** 12/11/2026  
**Aluno(a):** Gabriel Possa

## Introdução

A evolução dos modelos de linguagem está ligada a uma mudança importante na forma de processar sequências. Redes recorrentes (RNNs) mantêm um estado que avança token por token; Transformers usam atenção para relacionar posições da sequência em paralelo durante o treinamento. A arquitetura Transformer, apresentada em 2017, tornou mais simples aproveitar o paralelismo dos aceleradores e treinar modelos em grandes coleções de texto. Este relatório apresenta essa evolução, explica os componentes da atenção e descreve a construção e a execução local de LLMs.

## Parte 1 — Fundamentação teórica

### 1. O gargalo sequencial das RNNs

Em uma RNN, o estado oculto de uma posição depende do estado calculado na posição anterior. Uma forma simplificada de expressar isso é

\[
h_t = f(W_x x_t + W_h h_{t-1} + b),
\]

em que \(x_t\) é a entrada no instante \(t\), \(h_{t-1}\) é o estado anterior e \(h_t\) é o novo estado. Como \(h_t\) só pode ser calculado depois de \(h_{t-1}\), os passos da mesma sequência precisam ser executados em ordem. Isso limita o paralelismo ao longo do tempo: a GPU não pode calcular todos os estados da sequência simultaneamente.

Esse encadeamento pode deixar unidades de processamento ociosas, sobretudo quando a sequência é longa e cada operação por passo é relativamente pequena. O acelerador termina uma operação e precisa aguardar a dependência anterior antes de avançar. Ainda é possível paralelizar exemplos diferentes de um lote e operações internas de cada passo; portanto, a limitação não significa que uma RNN nunca use paralelismo. O problema estrutural é a dependência entre os passos temporais de uma mesma sequência, que reduz o aproveitamento do hardware e dificulta escalar o treinamento.

### 2. A mudança introduzida pelos Transformers

O artigo *Attention Is All You Need*, de Vaswani e colaboradores, apresentou o Transformer para tradução automática sem recorrer a camadas recorrentes ou convolucionais na arquitetura principal. Seu impacto foi mostrar que mecanismos de atenção, combinados com camadas posição a posição e codificação de posição, poderiam modelar relações entre palavras e treinar com alto paralelismo. O artigo relatou resultados competitivos ou superiores em tarefas de tradução com menor tempo de treinamento que sistemas comparados na época. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

Na autoatenção (*self-attention*), cada posição calcula quanto deve considerar as outras posições da sequência. Como as representações de entrada de todas as posições estão disponíveis juntas, os cálculos de atenção de uma camada podem ser organizados em operações matriciais paralelas. Isso substitui a cadeia recorrente de dependências temporais no processamento da camada. Para que a ordem não se perca, o Transformer incorpora informação posicional às representações.

Há uma ressalva importante: a geração de texto por um Transformer causal continua autorregressiva. Para produzir o próximo token, o modelo precisa conhecer os tokens anteriores; portanto, a geração de uma resposta também avança token a token. O grande paralelismo ocorre principalmente no treinamento e no processamento das posições já existentes, não na produção simultânea de todos os tokens futuros. A atenção padrão também tem custo quadrático em relação ao comprimento da sequência, o que motivou otimizações para contextos longos.

### 3. A mecânica de Query, Key e Value

Para cada vetor de entrada \(x_i\), o modelo calcula três projeções aprendidas:

\[
q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V.
\]

- **Query (consulta):** representa o que a posição atual procura nas demais posições. É comparada com as chaves.
- **Key (chave):** representa características usadas para indicar a relevância de uma posição para uma consulta.
- **Value (valor):** carrega a informação que será combinada quando uma posição recebe atenção.

A atenção escalada é calculada por

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V,
\]

em que \(QK^T\) produz os escores de compatibilidade entre consultas e chaves, \(d_k\) é a dimensão das chaves e a divisão por \(\sqrt{d_k}\) controla a escala dos escores. A máscara \(M\), quando necessária, impede que uma posição veja tokens futuros em um modelo causal. O *softmax* converte os escores em pesos; esses pesos formam uma soma ponderada dos valores \(V\). Em termos intuitivos: a consulta pergunta, as chaves ajudam a decidir onde olhar e os valores fornecem o conteúdo recuperado.

### 4. Etapas de construção de um LLM

1. **Tokenização e embeddings.** O texto é dividido em unidades chamadas tokens. Métodos subword, como BPE, aprendem combinações frequentes de caracteres ou unidades menores para representar palavras comuns com poucos tokens e palavras raras com segmentos reutilizáveis. Em seguida, cada identificador de token é mapeado para um vetor de embedding; informação posicional também é incorporada. [Sennrich, Haddow e Birch, 2015](https://arxiv.org/abs/1508.07909)
2. **Pré-treinamento.** O modelo é ajustado em grandes coleções de texto, que podem incluir fontes variadas e filtradas. Em um modelo causal, o objetivo usual é atribuir alta probabilidade ao próximo token dado o contexto anterior. Essa tarefa aprende padrões estatísticos e representações úteis, mas não garante compreensão humana, verdade ou intenção: a metáfora do “papagaio” destaca que fluência estatística não equivale, por si só, a compreensão fundamentada.
3. **Ajuste fino supervisionado (SFT).** O modelo recebe exemplos curados de instruções e respostas desejadas. O treinamento ensina o formato de diálogo e melhora a capacidade de seguir instruções. O trabalho sobre InstructGPT descreve uma etapa desse tipo antes do aprendizado com preferências humanas. [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)
4. **Alinhamento por preferências (RLHF ou DPO).** No RLHF clássico usado nesse fluxo, pessoas comparam respostas; essas preferências treinam um modelo de recompensa, e o modelo de linguagem é otimizado para obter recompensas maiores, normalmente com uma penalização para não se afastar demais do modelo de referência. DPO usa pares de respostas preferida e rejeitada para otimizar diretamente a política por uma função de perda derivada do modelo de preferências, evitando treinar e consultar um modelo de recompensa separado durante essa etapa. Os dois métodos dependem da qualidade e cobertura dos dados de preferência; nenhum garante alinhamento perfeito. [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)

## Parte 2 — Execução prática

### 1. Preparação com Ollama

O script entregue usa o endpoint HTTP local `http://localhost:11434/api/generate` e o modelo padrão `qwen2.5-coder:7b`. O Ollama precisa estar instalado e ativo, e o modelo deve estar disponível localmente (por exemplo, após `ollama pull qwen2.5-coder:7b`). Com o serviço iniciado, a inferência é feita na máquina local. O formato e a quantização efetivamente utilizados dependem da distribuição do modelo instalada pelo Ollama; verifique a versão/modelo com `ollama list` e `ollama show qwen2.5-coder:7b`.

### 2. Experimento com temperatura

Para comparar as configurações, o programa permite executar o mesmo prompt técnico em três temperaturas: `0.0`, `0.9` e `1.5`, mantendo `top_p` e o limite de geração iguais. Temperatura reescala os logits antes da amostragem: em geral, valores baixos concentram a distribuição nos tokens mais prováveis e valores mais altos a tornam mais dispersa. Com temperatura `0`, a intenção é escolher a opção mais provável (comportamento guloso/determinístico). Isso não garante resultados idênticos entre todas as versões, hardware ou caminhos de execução. `top_p` limita a amostragem ao conjunto menor de tokens cuja probabilidade acumulada alcança o limiar configurado.

| Cenário | Temperatura | Comportamento esperado | Registro desta execução |
|---|---:|---|---|
| A | 0.0 | Resposta mais previsível; adequada para observar uma solução técnica direta. | Gerou uma implementação direta com funções para ordenar e imprimir o vetor. A saída atingiu o limite de 300 tokens no meio da chamada `imprimir`. |
| B | 0.9 | Mais variação de formulação e escolhas, mantendo normalmente coerência. | Incluiu contadores de comparações e trocas e uma saída antecipada quando não há trocas. A saída atingiu 300 tokens antes de terminar o programa. |
| C | 1.5 | Maior aleatoriedade; aumenta a chance de respostas inconsistentes, erros ou desvios. | Usou uma função auxiliar de troca e uma estrutura diferente para ordenar e imprimir. A resposta permaneceu coerente, mas atingiu 300 tokens no meio da inicialização do vetor. |

O experimento foi executado localmente em 19/11/2026 com o modelo `qwen2.5-coder:7b`, `top_p=0.9` e `num_predict=300`, mantendo o mesmo prompt nos três cenários. As três respostas produziram abordagens diferentes e coerentes para o Bubble Sort; neste teste, a temperatura 1.5 não causou colapso sintático. Como todas as gerações chegaram ao limite de 300 tokens, nenhum dos trechos contém um programa completo. Isso limita a comparação da correção do código, mas não impede observar as diferenças de formulação e estrutura. A temperatura, isoladamente, não garante “caos”; a resposta também depende do modelo, prompt, `top_p`, seed e implementação de amostragem.

### 3–4. Integração Python, streaming e salvamento

O arquivo [`ollama_stream.py`](ollama_stream.py) envia uma requisição em streaming, imprime os fragmentos à medida que chegam e salva o texto completo em arquivo. Os parâmetros de temperatura, `top_p`, `num_predict`, endpoint e tempo limite podem ser alterados pela linha de comando. O padrão de `num_predict` foi elevado para 600 para reduzir cortes em respostas mais longas. Com `--experimento`, são feitas três chamadas com o mesmo prompt e temperaturas 0.0, 0.9 e 1.5; cada saída recebe um arquivo próprio. A implementação trata erros de conexão, tempo limite e respostas HTTP inválidas, e informa o motivo do término quando o servidor o fornece.

Comandos de exemplo:

```bash
python ollama_stream.py --temperature 0.0 --top-p 0.9 --num-predict 600 --output resposta.txt
python ollama_stream.py --experimento --top-p 0.9 --num-predict 300
```

O segundo comando cria `experimento_temp_0.0.txt`, `experimento_temp_0.9.txt` e `experimento_temp_1.5.txt`. Os resultados e o desempenho dependem do modelo realmente instalado e do hardware local.

## Conclusão

As RNNs processam cada sequência com dependências entre passos, enquanto os Transformers permitem calcular as relações de atenção entre posições em paralelo durante o treinamento. A atenção Q/K/V fornece um mecanismo para ponderar informação contextual; pré-treinamento, SFT e alinhamento por preferências cumprem papéis distintos na construção de um assistente. O script implementa a integração local pedida e automatiza a comparação entre temperaturas. O experimento foi executado com três temperaturas, e as respostas foram salvas nos arquivos de saída. Os resultados mostram variações estruturais sem o colapso esperado no cenário de temperatura 1.5; o limite configurado, porém, interrompeu todos os códigos antes de sua conclusão.

## Referências

- VASWANI, A. et al. *Attention Is All You Need*. 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- SENNRICH, R.; HADDOW, B.; BIRCH, A. *Neural Machine Translation of Rare Words with Subword Units*. 2015. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
- OUYANG, L. et al. *Training language models to follow instructions with human feedback*. 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155).
- RAFAILOV, R. et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
