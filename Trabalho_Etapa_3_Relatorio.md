# Trabalho 3ª Etapa — Inteligência Artificial

**Centro Universitário Presidente Antônio Carlos — UNIPAC Barbacena**  
**Curso:** Ciência da Computação  
**Disciplina:** Inteligência Artificial  
**Professor:** Felipe Roncalli de Paula Carneiro  
**Data:** 20/11/2026  
**Aluno(a):** Gabriel Possa  

---

## Introdução

A evolução do Processamento de Linguagem Natural (PLN) e dos Modelos de Linguagem de Grande Escala (LLMs) está fundamentada em uma profunda revolução arquitetural. Historicamente, as Redes Neurais Recorrentes (RNNs) dominaram o tratamento de dados sequenciais mantendo estados ocultos propagados passo a passo no tempo. Contudo, suas restrições intrínsecas de paralelização e a incapacidade de explorar a plenitude do hardware moderno motivaram o surgimento da arquitetura Transformer em 2017.

Baseado inteiramente no mecanismo de atenção (especificamente *Self-Attention* com formulação Query, Key e Value), o Transformer viabilizou o treinamento massivo e distribuído sobre gigantescas bases textuais. Este trabalho aborda os conceitos teóricos essenciais dessa transição e documenta um laboratório prático individual com inferência local desconectada via framework Ollama, explorando a quantização em formato GGUF, automação com requisições em fluxo contínuo (*streaming*) em Python e a análise estocástica sob diferentes hiperparâmetros de temperatura e amostragem (*Top-P*).

---

## Parte 1 — Atividade de Pesquisa e Investigação (Fundamentação Teórica)

### 1. O Gargalo Sequencial (RNNs)

#### Processamento Serial e Falha Estrutural na Paralelização
As Redes Neurais Recorrentes (RNNs) foram concebidas sob a premissa de processamento estritamente sequencial. A equação fundamental que governa a atualização do vetor de estado oculto em cada instante de tempo \(t\) é expressa por:

\[
h_t = f(W_x x_t + W_h h_{t-1} + b)
\]

onde \(x_t \in \mathbb{R}^d\) representa o vetor de entrada no passo atual, \(h_{t-1} \in \mathbb{R}^h\) é o estado oculto proveniente do instante anterior, \(W_x\) e \(W_h\) são as matrizes de pesos aprendíveis, \(b\) é o vetor de viés e \(f\) é uma função de ativação não linear (comumente \(\tanh\) ou sigmoid).

A dependência temporal explícita de \(h_t\) em relação a \(h_{t-1}\) impõe uma restrição de ordem estrita: o cálculo do estado no passo \(t\) não pode iniciar antes que o cálculo do passo \(t-1\) seja concluído e disponibilizado. Consequentemente, para uma sequência de comprimento \(T\), o grafo computacional exige obrigatoriamente \(T\) passos sequenciais. Mesmo em variantes mais sofisticadas, como LSTMs (*Long Short-Term Memory*) e GRUs (*Gated Recurrent Units*), os mecanismos de *gates* (portões de esquecimento, entrada e saída) operam sobre recorrências encadeadas, preservando intacto esse gargalo serial inegociável.

#### Ociosidade de Hardware nas GPUs Modernas
Os aceleradores gráficos modernos (GPUs e TPUs) são processadores massivamente paralelos, compostos por milhares de núcleos de processamento aritmético (por exemplo, núcleos CUDA e Tensor Cores da arquitetura NVIDIA) projetados para maximizar a vazão (*throughput*) de operações tensoriais e multiplicações matriciais densas em larga escala. 

Em uma RNN, embora seja viável paralelizar exemplos distintos dentro de um *batch* e as multiplicações internas das projeções de entrada de um único passo, a dimensão temporal não pode ser distribuída. Isso gera um padrão operacional de "baixa densidade aritmética": o acelerador executa uma operação relativamente pequena, precisa sincronizar a memória para passar o vetor \(h_t\) para a iteração seguinte e aguarda o próximo ciclo. Durante esse intervalo temporal de latência de sincronização e dependência estrita de dados, a grande maioria dos núcleos da GPU permanece ociosa, degradando o aproveitamento computacional (*low GPU utilization*) e tornando o pré-treinamento em corpora de centenas de bilhões de tokens impraticável em larga escala.

---

### 2. A Quebra de Paradigma (Transformers)

#### A Revolução de 2017: *Attention Is All You Need*
Em 2017, o artigo seminal *Attention Is All You Need* (Vaswani et al.) transformou radicalmente a computação de IA ao eliminar completamente qualquer camada recorrente ou convolucional na estrutura primária do modelo. O trabalho introduziu a arquitetura **Transformer**, demonstrando que o mecanismo de atenção puro é suficiente para capturar dependências de longo alcance entre todas as posições de uma sequência, independentemente da distância entre elas.

O impacto dessa quebra de paradigma na engenharia de inteligência artificial foi imediato: ao transferir o custo computacional de laços temporais sequenciais para grandes operações de álgebra linear matricial (\(QK^T\)), tornou-se possível alimentar modelos com sequências completas de uma só vez. A capacidade de saturação dos clusters de GPUs escalou de forma linear, viabilizando o surgimento dos modelos fundacionais modernos (GPT, BERT, Llama, Qwen, Claude, Gemini).

#### Abandono da Recorrência e Processamento Simultâneo
No Transformer, a recorrência linear é substituída pelo mecanismo de **Self-Attention** (Autoatenção). Em vez de esperar o fluxo temporal passar de palavra em palavra, todas as palavras da sentença são projetadas simultaneamente em matrizes na memória. Cada token calcula pontuações de similaridade contextual com todos os outros tokens da frase em uma única operação matricial paralelizada.

Como o mecanismo de autoatenção não possui naturalmente a noção de ordenação cronológica (ele é invariante a permutações), o Transformer incorpora vetores de **Codificação Posicional** (*Positional Encodings* estáticos senoidais ou aprendidos/rotacionais como RoPE) aos embeddings iniciais.

> **Ressalva importante sobre Inferência:** O processamento perfeitamente paralelo ocorre durante o **treinamento** e na fase de processamento do prompt (*prefill*). Na etapa de geração autoregressiva (*decoding*), o modelo causal necessita produzir um token por vez para anexá-lo ao contexto antes de prever o próximo. Entretanto, mesmo durante a inferência, a arquitetura utiliza estratégias como o *KV Cache* para evitar recalcular os estados dos tokens anteriores, mantendo eficiência muito superior à das RNNs.

---

### 3. A Mecânica do Sistema de Busca (Query, Key, Value)

A mecânica de atenção opera analogamente a um sistema de recuperação de informação (*retrieval*) vetorial contínuo e diferenciável. Para cada vetor de token \(x_i \in \mathbb{R}^{d_{model}}\), são geradas três projeções lineares através de matrizes de pesos treináveis \(W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}\):

\[
q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V
\]

* **Query (Consulta - \(Q\)):** Representa o vetor de busca da palavra atual. Codifica o que o token no passo em questão está ativamente procurando no contexto restante da frase para enriquecer seu próprio significado.
* **Key (Chave - \(K\)):** Representa as etiquetas identificadoras e descritivas que cada palavra da sequência oferece. Serve como índice para comparação de compatibilidade semântica com a Query.
* **Value (Valor - \(V\)):** Representa o conteúdo semântico real e a informação vetorial que será extraída e combinada ponderadamente caso haja correspondência entre Query e Key.

#### Formulação Matemática da Atenção Escalada
A atenção entre todas as consultas \(Q\), chaves \(K\) e valores \(V\) da sequência é computada matricialmente por:

\[
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
\]

1. **Produto Escalar (\(Q K^T\)):** Mede a afinidade/relevância entre cada consulta e todas as chaves da sequência, resultando em uma matriz de pontuações quadrática \(N \times N\).
2. **Fator de Escala (\(\frac{1}{\sqrt{d_k}}\)):** Para dimensões elevadas de \(d_k\), o produto escalar atinge magnitudes altas, o que levaria a função softmax para regiões de gradiente quase nulo (saturação dos gradientes). A divisão por \(\sqrt{d_k}\) normaliza a variância dos escores para 1, estabilizando o treinamento.
3. **Máscara Causal (\(M\)):** Em modelos autorregressivos (decoders de linguagem), \(M\) atribui \(-\infty\) às posições futuras, garantindo que o token no índice \(i\) só consulte tokens nos índices \(j \le i\).
4. **Softmax e Ponderação:** O softmax transforma os escores em uma distribuição discreta de probabilidades (pesos de atenção que somam 1), multiplicando-os pela matriz de valores \(V\). O resultado é uma combinação linear dos vetores de valor, focando no que é contextualmente relevante.

---

### 4. A Linha de Montagem de um LLM

A construção industrial de um LLM segue um pipeline rigoroso dividido em quatro estágios complementares:

```
[Texto Bruto] 
     │
     ▼
1. Tokenização & Embeddings (BPE, Vocabulário, Espaço Vetorial)
     │
     ▼
2. Pré-Treinamento Massivo (Self-Supervised, Previsão do Próximo Token, "Papagaio Estocástico")
     │
     ▼
3. Ajuste Fino Supervisionado - SFT (Instrução, Formato Assistente, Diálogos Curados)
     │
     ▼
4. Alinhamento de Preferências (RLHF com PPO vs. Otimização Direta DPO)
     │
     ▼
[LLM Alinhado e Seguro para Produção]
```

#### 1. Tokenização e Embeddings
Textos em caracteres brutos são segmentados em unidades atômicas (tokens) utilizando algoritmos baseados em frequência de subpalavras, principalmente **BPE (Byte Pair Encoding)**. O BPE compacta palavras comuns em um único token e decompõe palavras raras ou técnicas em subunidades reutilizáveis, equilibrando a cobertura de vocabulário e a eficiência computacional. Em seguida, cada identificador numérico de token é indexado em uma tabela de *embeddings*, sendo mapeado para vetores densos em um espaço latente de alta dimensionalidade (ex: 3584 dimensões no Qwen 7B), acrescido das informações de posição.

#### 2. Pré-treinamento (O Paradoxo do Papagaio)
Nesta fase, o modelo processa trilhões de tokens de dados públicos e curados (livros, código-fonte, páginas da web filtradas). O objetivo de perda é puramente estatístico: maximizar a verossimilhança do próximo token dado o histórico anterior (modelagem causal auto-supervisionada):

\[
\mathcal{L}_{pretrain} = -\sum_{t} \log P(x_t \mid x_{<t}; \theta)
\]

Esse treinamento gera um previsor de probabilidades altamente sofisticado, capaz de capturar sintaxe, padrões lógicos, fatos e código. No entanto, o modelo base sofre do chamado **"paradoxo do papagaio estocástico"**: ele repete e recombina padrões estatísticos com impressionante fluência formal, mas sem fundamentação empírica de verdade, intenção consciente ou senso crítico inerente.

#### 3. Ajuste Fino Supervisionado (SFT - *Supervised Fine-Tuning*)
O modelo pré-treinado tende a completar texto livremente em vez de responder prontamente a comandos. No SFT, ele é ajustado em milhares de pares curados de instruções e respostas de alta qualidade no formato de diálogo (`<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>`). Esta fase ensina o modelo a reconhecer seu papel de assistente, estruturar respostas concisas e seguir instruções lógicas e de programação com obediência.

#### 4. Alinhamento (RLHF vs. DPO)
Para garantir que o assistente seja útil, inofensivo e honesto (*Helpful, Harmless, Honest*), aplica-se o alinhamento de preferências humanas:
* **RLHF (Reinforcement Learning from Human Feedback):** Treina primeiramente um *Modelo de Recompensa* (Reward Model) baseado em preferências humanas comparativas. Em seguida, otimiza o LLM via algoritmos de reforço (geralmente PPO), utilizando uma penalidade de divergência KL em relação ao modelo original para evitar o colapso de política.
* **DPO (Direct Preference Optimization):** Elimina a necessidade de treinar um modelo de recompensa separado e a instabilidade do PPO. O DPO deriva matematicamente uma função de perda analítica que otimiza os pesos do próprio LLM diretamente sobre pares de respostas preferidas (\(y_w\)) e rejeitadas (\(y_l\)), proporcionando convergência mais estável, rápida e de menor custo computacional.

---

## Parte 2 — Atividade de Execução Prática

### 1. Setup do Ambiente com Ollama

O ambiente foi configurado localmente utilizando o framework **Ollama** para execução direta de modelos em formato **GGUF** (*GPT-Generated Unified Format*).

* **Modelo selecionado:** `qwen2.5-coder:7b` (otimizado para tarefas avançadas de código e raciocínio lógico).
* **Parâmetros e Quantização:**
  * Arquitetura: `qwen2`
  * Contagem de parâmetros: `7.6B` (7,61 bilhões)
  * Nível de quantização: `Q4_K_M` (pesos quantizados em 4 bits com matrizes críticas preservadas)
  * Janela de contexto nativa: `32.768 tokens`
  * Tamanho do arquivo em disco: `4,68 GB`
* **Garantia de Execução Desconectada:** A inferência é realizada integralmente no hardware local da máquina (`dp-550xda`), comunicando-se via loopback TCP (`http://127.0.0.1:11434`), garantindo privacidade estrita, ausência de latência de rede externa e isenção de filtros corporativos de nuvem.

Inspeção direta dos metadados do modelo via CLI do Ollama:

```text
$ ollama show qwen2.5-coder:7b
  Model
    architecture        qwen2          
    parameters          7.6B           
    context length      32768          
    embedding length    3584           
    quantization        Q4_K_M         
    template source     go_template    

  Capabilities
    completion    
    tools         
    insert        

  System
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

---

### 2. Experimento de Hiperparâmetros (Temperatura e Amostragem)

O experimento prático submeteu rigorosamente o **mesmo prompt técnico** exigindo lógica de programação:

> *"Escreva um algoritmo de ordenação Bubble Sort em C bem documentado."*

Foram configurados três cenários para avaliar a influência da temperatura (\(T\)) na distribuição softmax sobre os logits dos tokens:

\[
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]

A análise empírica comparou o impacto de \(T\) com um teto suficiente de geração (`num_predict = 600`), permitindo observar o código completo e a manifestação estocástica real do motor:

| Cenário | Configuração | Comportamento Teórico | Registro da Execução Empírica |
|---|:---:|---|---|
| **A** | **T = 0.0**<br>`top_p = 0.9`<br>`tokens: 600` | **Determinístico:** Logits colapsam no token de probabilidade máxima (*greedy decoding*). Ideal para sintaxe estrita e matemática. | **Execução perfeita e rigorosa:** Produziu um código C completo, canônico e idiomático com funções `imprimirArray`, `bubbleSort`, `main` e `return 0;`. Explicou passo a passo a complexidade temporal e a lógica de trocas sem nenhuma falha sintática. |
| **B** | **T = 0.9**<br>`top_p = 0.9`<br>`tokens: 407` | **Criatividade controlada:** Favorece fluidez e variabilidade léxica mantendo coerência via corte de probabilidade acumulada (*nucleus*). | **Execução fluida e natural:** O modelo adotou uma abordagem arquitetural diferente: criou uma função auxiliar `swap(int *xp, int *yp)` com ponteiros, implementou comentários claros e encerrou a geração naturalmente com sinal de parada (`done: true`, motivo `stop`) aos 407 tokens. |
| **C** | **T = 1.5**<br>`top_p = 1.0`<br>`tokens: 600` | **O Caos:** Achata a distribuição de probabilidade, elevando o peso relativo de tokens improváveis da cauda longa. | **Colapso sintático e alucinações severas:** O fenômeno do caos manifestou-se plenamente em múltiplas frentes:<br>1. *Desvios léxicos da cauda:* denominou Bubble Sort como ordenação por "percolação" e gerou arrays com "valores unirands".<br>2. *Mistura de idiomas e falsos cognatos:* escreveu "Ela continua this processo" e traduziu *sorted* erroneamente para "lista estará sorteada".<br>3. *Alucinação de comentários:* descreveu flags de otimização que não existiam no código.<br>4. *Colapso sintático grave em C:* chamou `troca('&a[j]')` passando string entre aspas simples para função com dois ponteiros, em seguida `troca(&a[j+1])` com argumento único, inventando justificativas absurdas nos comentários. |

---

### 3. Automação de Inferência via Python (Streaming) e Desafio de Código

Para atender aos requisitos de integração e ao desafio prático de código, o script [`ollama_stream.py`](file:///home/dp/Trabalho-IA-3Etapa/ollama_stream.py) foi implementado em Python puro (utilizando a biblioteca `requests`), conectando-se ao endpoint local `http://127.0.0.1:11434/api/generate`.

#### Funcionalidades Implementadas:
1. **Processamento Autorregressivo com Streaming em Tempo Real:**
   * Utiliza `stream=True` na requisição HTTP `POST`.
   * Itera sobre o fluxo de resposta contínuo linha a linha com `response.iter_lines(decode_unicode=True)`.
   * Deserializa os fragmentos JSON (`json.loads(line)`) e envia os tokens instantaneamente para a saída padrão (`sys.stdout`) com `flush=True`, eliminando qualquer atraso de buffering visual.
2. **Desafio de Controle Dinâmico de Hiperparâmetros:**
   * Parâmetros `--temperature`, `--top-p` e `--num-predict` integrados via `argparse` com validação de faixas e finitude numérica.
   * Suporte a controle de tempo limite (`--timeout`) e seleção de modelo (`--model`).
3. **Persistência Automática em Arquivo de Texto:**
   * Grava progressivamente os fragmentos no arquivo `.txt` definido via `--output` ou gera os arquivos independentes do experimento (`experimento_temp_0.0.txt`, `experimento_temp_0.9.txt`, `experimento_temp_1.5.txt`) através da flag `--experimento`.

#### Exemplo de Invocação e Teste:
```bash
# Execução individual com parâmetros dinâmicos e persistência em arquivo
python3 ollama_stream.py --temperature 0.0 --top-p 0.9 --num-predict 600 --output experimento_temp_0.0.txt

# Execução do Cenário B (T = 0.9)
python3 ollama_stream.py --temperature 0.9 --top-p 0.9 --num-predict 600 --output experimento_temp_0.9.txt

# Execução do Cenário C com cauda longa irrestrita (T = 1.5 e Top-P = 1.0)
python3 ollama_stream.py --temperature 1.5 --top-p 1.0 --num-predict 600 --output experimento_temp_1.5.txt
```

---

## Conclusão

Este trabalho investigou de forma integral a transição fundamental do processamento sequencial de RNNs para o paralelismo massivo dos Transformers. Ficou evidente que o gargalo temporal das RNNs (\(h_t = f(h_{t-1})\)) tornava impraticável a saturação eficiente do hardware de aceleração gráfica em larga escala. O mecanismo de autoatenção escalada baseado em Query, Key e Value resolveu esse entrave estrutural no treinamento, permitindo que os estágios modernos de industrialização de LLMs (Tokenização/Embeddings, Pré-Treinamento massivo, SFT e Alinhamento via RLHF/DPO) produzissem modelos de elevada capacidade cognitiva.

No âmbito prático, a execução local e desconectada do modelo `qwen2.5-coder:7b` (quantizado em `Q4_K_M` no padrão GGUF) via framework Ollama comprovou a viabilidade de inferência soberana e privada em computadores de uso pessoal. O script em Python supriu com êxito os requisitos de streaming de tokens em tempo real e controle dinâmico de amostragem. Por fim, a avaliação empírica das três temperaturas validou as previsões teóricas: a temperatura nula garantiu precisão matemática determinística, a temperatura intermediária produziu código alternativo elegante com terminação natural, e a temperatura extrema com cauda longa aberta evidenciou com clareza o colapso sintático e as alucinações léxicas típicas da dispersão estocástica.

---

## Referências Bibliográficas

* VASWANI, A.; SHAZEER, N.; PARMAR, N.; USZKOREIT, J.; JONES, L.; GOMEZ, A. N.; KAISER, L.; POLOSUKHIN, I. **Attention Is All You Need**. Advances in Neural Information Processing Systems (NeurIPS), 2017. Disponível em: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
* SENNRICH, R.; HADDOW, B.; BIRCH, A. **Neural Machine Translation of Rare Words with Subword Units**. Association for Computational Linguistics (ACL), 2015. Disponível em: [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
* OUYANG, L. et al. **Training language models to follow instructions with human feedback** (InstructGPT). Advances in Neural Information Processing Systems (NeurIPS), 2022. Disponível em: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155).
* RAFAILOV, R.; SHARMA, A.; MITCHELL, E.; ERMON, S.; MANNING, C. D.; FINN, C. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. Advances in Neural Information Processing Systems (NeurIPS), 2023. Disponível em: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
