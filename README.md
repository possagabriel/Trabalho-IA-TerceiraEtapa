# Trabalho 3ª Etapa — Inteligência Artificial

**Centro Universitário Presidente Antônio Carlos — UNIPAC Barbacena**  
**Curso:** Ciência da Computação  
**Disciplina:** Inteligência Artificial  
**Professor:** Felipe Roncalli de Paula Carneiro  
**Aluno:** Gabriel Possa  

Projeto acadêmico e laboratório individual sobre a evolução arquitetural desde as Redes Neurais Recorrentes (RNNs) até os Modelos de Linguagem de Grande Escala (Transformers / LLMs), complementado por um experimento prático local de geração de código via framework Ollama.

---

## Conteúdo do Repositório

- **[Trabalho_Etapa_3_Relatorio.pdf](Trabalho_Etapa_3_Relatorio.pdf)**: Entregável final formal em formato PDF diagramado conforme os padrões acadêmicos.
- **[Trabalho_Etapa_3_Relatorio.md](Trabalho_Etapa_3_Relatorio.md)**: Relatório dissertativo estruturado contendo a Fundamentação Teórica (Parte 1) e a Análise da Execução Prática (Parte 2).
- **[ollama_stream.py](ollama_stream.py)**: Script em Python para inferência com streaming de tokens em tempo real (`stream=True`), controle dinâmico de `temperature`, `top_p` e `num_predict`, e gravação automática em arquivo `.txt`.
- **`experimento_temp_*.txt`**: Registros dos testes empíricos com o mesmo prompt técnico sob três configurações:
  - `experimento_temp_0.0.txt`: Cenário A ($T = 0.0$, totalmente determinístico, código C completo e canônico).
  - `experimento_temp_0.9.txt`: Cenário B ($T = 0.9$, criatividade controlada, função `swap` com ponteiros, término natural com `stop`).
  - `experimento_temp_1.5.txt`: Cenário C ($T = 1.5$, $Top-P = 1.0$, manifestação do caos estocástico, alucinações léxicas e colapso sintático em C).
- **[build_report_pdf.py](build_report_pdf.py)**: Utilitário para compilar o relatório Markdown em HTML e exportar para PDF via LibreOffice.
- **[requirements.txt](requirements.txt)**: Dependências Python do projeto.

---

## Pré-requisitos e Instalação

1. **Python 3.9+**
2. **Ollama**:
   No Linux, instale através do comando oficial:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. **Download do Modelo**:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
4. **Dependências Python**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Como Executar

### 1. Inferência Individual com Streaming
```bash
python3 ollama_stream.py --temperature 0.2 --top-p 0.9 --num-predict 600 --output resposta_ollama.txt
```

### 2. Execução dos Experimentos de Temperatura
Para executar os três cenários com 600 tokens:
```bash
# Cenário A: T = 0.0 (Determinístico)
python3 ollama_stream.py --temperature 0.0 --top-p 0.9 --num-predict 600 --output experimento_temp_0.0.txt

# Cenário B: T = 0.9 (Fluido / Criatividade Controlada)
python3 ollama_stream.py --temperature 0.9 --top-p 0.9 --num-predict 600 --output experimento_temp_0.9.txt

# Cenário C: T = 1.5 (O Caos com cauda longa aberta)
python3 ollama_stream.py --temperature 1.5 --top-p 1.0 --num-predict 600 --output experimento_temp_1.5.txt
```

### 3. Recompilar o Relatório em PDF
```bash
python3 build_report_pdf.py
```
O arquivo `Trabalho_Etapa_3_Relatorio.pdf` será gerado/atualizado na raiz do projeto.
