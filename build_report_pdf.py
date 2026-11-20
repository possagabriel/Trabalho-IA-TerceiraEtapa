#!/usr/bin/env python3
"""Converte o relatório Markdown em um PDF acadêmico de alta qualidade e formatação impecável."""

import html
import re
import subprocess
from pathlib import Path
from markdown_it import MarkdownIt

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "Trabalho_Etapa_3_Relatorio.md"
HTML_PATH = BASE_DIR / "Trabalho_Etapa_3_Relatorio.html"
PDF_PATH = BASE_DIR / "Trabalho_Etapa_3_Relatorio.pdf"


def replace_math_before_markdown(text: str) -> str:
    """Substitui blocos LaTeX por marcadores seguros antes do processamento do Markdown."""
    # Bloco h_t
    text = re.sub(
        r'\\\[\s*h_t\s*=\s*f\(W_x\s*x_t\s*\+\s*W_h\s*h_\{t-1\}\s*\+\s*b\)\s*\\\]',
        '<div class="math-block"><b><i>h<sub>t</sub></i> = <i>f</i>(<i>W<sub>x</sub></i> <i>x<sub>t</sub></i> + <i>W<sub>h</sub></i> <i>h<sub>t−1</sub></i> + <i>b</i>)</b></div>',
        text
    )
    # Bloco q_i, k_i, v_i
    text = re.sub(
        r'\\\[\s*q_i\s*=\s*x_i\s*W_Q,\s*\\qquad\s*k_i\s*=\s*x_i\s*W_K,\s*\\qquad\s*v_i\s*=\s*x_i\s*W_V\.?\s*\\\]',
        '<div class="math-block"><b><i>q<sub>i</sub></i> = <i>x<sub>i</sub></i> <i>W<sub>Q</sub></i> &emsp;&emsp; <i>k<sub>i</sub></i> = <i>x<sub>i</sub></i> <i>W<sub>K</sub></i> &emsp;&emsp; <i>v<sub>i</sub></i> = <i>x<sub>i</sub></i> <i>W<sub>V</sub></i></b></div>',
        text
    )
    # Bloco Attention
    text = re.sub(
        r'\\\[\s*\\operatorname\{Attention\}\(Q,\s*K,\s*V\)\s*=\s*\\operatorname\{softmax\}\\!\\left\(\\frac\{Q\s*K\^T\}\{\\sqrt\{d_k\}\}\s*\+\s*M\\right\)V\.?\s*\\\]',
        '<div class="math-block"><b>Attention(<i>Q, K, V</i>) = softmax( (<i>Q K</i><sup>T</sup> / &radic;<i>d<sub>k</sub></i>) + <i>M</i> ) <i>V</i></b></div>',
        text
    )
    # Bloco Loss pretraining
    text = re.sub(
        r'\\\[\s*\\mathcal\{L\}_\{pretrain\}\s*=\s*-\\sum_\{t\}\s*\\log\s*P\(x_t\s*\\mid\s*x_<t;\s*\\theta\)\s*\\\]',
        '<div class="math-block"><b><i>L</i><sub>pretrain</sub> = &minus; &sum;<sub><i>t</i></sub> log <i>P</i>(<i>x<sub>t</sub></i> | <i>x</i><sub>&lt;<i>t</i></sub>; &theta;)</b></div>',
        text
    )
    # Bloco softmax temperatura
    text = re.sub(
        r'\\\[\s*P\(x_i\)\s*=\s*\\frac\{\\exp\(z_i\s*/\s*T\)\}\{\\sum_j\s*\\exp\(z_j\s*/\s*T\)\}\s*\\\]',
        '<div class="math-block"><b><i>P</i>(<i>x<sub>i</sub></i>) = exp(<i>z<sub>i</sub></i> / <i>T</i>) / &sum;<sub><i>j</i></sub> exp(<i>z<sub>j</sub></i> / <i>T</i>)</b></div>',
        text
    )

    # Inline math substituições seguras
    replacements = [
        (r'\\\(h_t\\\)', r'<i>h<sub>t</sub></i>'),
        (r'\\\(h_\{t-1\}\\\)', r'<i>h<sub>t−1</sub></i>'),
        (r'\\\(x_t\\\)', r'<i>x<sub>t</sub></i>'),
        (r'\\\(W_x\\\)', r'<i>W<sub>x</sub></i>'),
        (r'\\\(W_h\\\)', r'<i>W<sub>h</sub></i>'),
        (r'\\\(b\\\)', r'<i>b</i>'),
        (r'\\\(f\\\)', r'<i>f</i>'),
        (r'\\\(T\\\)', r'<i>T</i>'),
        (r'\\\(x_i\\\)', r'<i>x<sub>i</sub></i>'),
        (r'\\\(q_i\\\)', r'<i>q<sub>i</sub></i>'),
        (r'\\\(k_i\\\)', r'<i>k<sub>i</sub></i>'),
        (r'\\\(v_i\\\)', r'<i>v<sub>i</sub></i>'),
        (r'\\\(Q\\\)', r'<i>Q</i>'),
        (r'\\\(K\\\)', r'<i>K</i>'),
        (r'\\\(V\\\)', r'<i>V</i>'),
        (r'\\\(W_Q\\\)', r'<i>W<sub>Q</sub></i>'),
        (r'\\\(W_K\\\)', r'<i>W<sub>K</sub></i>'),
        (r'\\\(W_V\\\)', r'<i>W<sub>V</sub></i>'),
        (r'\\\(QK\^T\\\)', r'<i>Q K</i><sup>T</sup>'),
        (r'\\\(Q K\^T\\\)', r'<i>Q K</i><sup>T</sup>'),
        (r'\\\(d_k\\\)', r'<i>d<sub>k</sub></i>'),
        (r'\\\(d_\{model\}\\\)', r'<i>d</i><sub>model</sub>'),
        (r'\\\(M\\\)', r'<i>M</i>'),
        (r'\\\(y_w\\\)', r'<i>y<sub>w</sub></i>'),
        (r'\\\(y_l\\\)', r'<i>y<sub>l</sub></i>'),
        (r'\\\(x_t \\in \\mathbb\{R\}\^d\\\)', r'<i>x<sub>t</sub></i> &isin; &#8477;<sup><i>d</i></sup>'),
        (r'\\\(h_\{t-1\} \\in \\mathbb\{R\}\^h\\\)', r'<i>h<sub>t−1</sub></i> &isin; &#8477;<sup><i>h</i></sup>'),
        (r'\\\(W_Q, W_K, W_V \\in \\mathbb\{R\}\^\{d_\{model\} \\times d_k\}\\\)', r'<i>W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub></i> &isin; &#8477;<sup><i>d</i><sub>model</sub> &times; <i>d<sub>k</sub></i></sup>'),
        (r'\\\(x_i \\in \\mathbb\{R\}\^\{d_\{model\}\}\\\)', r'<i>x<sub>i</sub></i> &isin; &#8477;<sup><i>d</i><sub>model</sub></sup>'),
        (r'\\\(N \\times N\\\)', r'<i>N</i> &times; <i>N</i>'),
        (r'\\\(-\\infty\\\)', r'&minus;&infin;'),
        (r'\\\(j \\le i\\\)', r'<i>j</i> &le; <i>i</i>'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    return text


def build_html() -> str:
    raw_md = MD_PATH.read_text(encoding="utf-8")

    # Remove o bloco de título inicial do Markdown para evitar duplicação com o cabeçalho institucional estilizado
    # Remove de "# Trabalho 3ª Etapa" até antes de "## Introdução"
    cleaned_md = re.sub(
        r'^#\s+Trabalho 3ª Etapa.*?(?=##\s+Introdução)',
        '',
        raw_md,
        flags=re.DOTALL
    )

    cleaned_md = replace_math_before_markdown(cleaned_md)

    parser = MarkdownIt("commonmark").enable("table")
    body_html = parser.render(cleaned_md)

    full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Trabalho 3ª Etapa — Inteligência Artificial — UNIPAC Barbacena</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 20mm 18mm 20mm 22mm;
  }}

  body {{
    font-family: 'Liberation Serif', 'DejaVu Serif', 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.4;
    color: #111111;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }}

  /* Cabeçalho Oficial */
  .institution-box {{
    border: 2px solid #003366;
    border-radius: 4px;
    background-color: #f7f9fc;
    padding: 10px 14px;
    margin-bottom: 20px;
    page-break-inside: avoid;
  }}
  .inst-name {{
    font-size: 13pt;
    font-weight: bold;
    color: #003366;
    text-align: center;
    text-transform: uppercase;
    margin: 0 0 2px 0;
  }}
  .inst-course {{
    font-size: 10pt;
    font-weight: 600;
    color: #333333;
    text-align: center;
    margin: 0 0 8px 0;
  }}
  .inst-title {{
    font-size: 12pt;
    font-weight: bold;
    color: #8b0000;
    text-align: center;
    text-transform: uppercase;
    border-top: 1px solid #ccd5e0;
    border-bottom: 1px solid #ccd5e0;
    padding: 4px 0;
    margin: 6px 0;
  }}
  .info-table {{
    width: 100%;
    margin-top: 6px;
    font-size: 9.5pt;
    border-collapse: collapse;
  }}
  .info-table td {{
    padding: 2px 4px;
    border: none;
  }}

  /* Títulos */
  h2 {{
    font-size: 12.5pt;
    font-weight: bold;
    color: #003366;
    border-bottom: 1.5px solid #003366;
    padding-bottom: 3px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 11pt;
    font-weight: bold;
    color: #1a202c;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 10.5pt;
    font-weight: bold;
    color: #2d3748;
    margin-top: 12px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}

  /* Parágrafos e Listas */
  p {{
    text-align: justify;
    text-indent: 1.0cm;
    margin-top: 0;
    margin-bottom: 7px;
  }}
  li p {{
    text-indent: 0;
    margin-bottom: 3px;
  }}
  ul, ol {{
    margin-top: 2px;
    margin-bottom: 8px;
    padding-left: 24px;
    text-align: justify;
  }}
  li {{
    margin-bottom: 3px;
  }}

  /* Bloco de Citação */
  blockquote {{
    border-left: 4px solid #003366;
    background-color: #f0f4f8;
    margin: 10px 0;
    padding: 6px 12px;
    font-size: 9.5pt;
  }}
  blockquote p {{
    text-indent: 0;
  }}

  /* Tabelas */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9pt;
    page-break-inside: avoid;
  }}
  table, th, td {{
    border: 1px solid #4a5568;
  }}
  th {{
    background-color: #003366;
    color: #ffffff;
    font-weight: bold;
    padding: 6px 8px;
    text-align: center;
  }}
  td {{
    padding: 6px 8px;
    vertical-align: top;
    text-align: left;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  /* Blocos de Código */
  pre {{
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 8px 10px;
    font-family: 'Liberation Mono', 'DejaVu Sans Mono', monospace;
    font-size: 8pt;
    line-height: 1.3;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 10px 0;
  }}
  code {{
    font-family: 'Liberation Mono', 'DejaVu Sans Mono', monospace;
    font-size: 8.5pt;
    background-color: #e2e8f0;
    padding: 1px 3px;
    border-radius: 2px;
  }}
  pre code {{
    background-color: transparent;
    padding: 0;
  }}

  /* Equações Matemáticas */
  .math-block {{
    text-align: center;
    background-color: #f8fafc;
    border: 1px dashed #94a3b8;
    padding: 6px 8px;
    margin: 10px auto;
    font-size: 10.5pt;
    font-family: 'Liberation Serif', serif;
    page-break-inside: avoid;
  }}

  hr {{
    border: 0;
    border-top: 1px solid #cbd5e1;
    margin: 18px 0;
  }}

  .footer-sig {{
    font-size: 8.5pt;
    color: #64748b;
    text-align: center;
    margin-top: 25px;
    border-top: 1px solid #e2e8f0;
    padding-top: 6px;
  }}
</style>
</head>
<body>

<div class="institution-box">
  <div class="inst-name">Centro Universitário Presidente Antônio Carlos — UNIPAC Barbacena</div>
  <div class="inst-course">Curso de Bacharelado em Ciência da Computação &bull; Disciplina de Inteligência Artificial</div>
  <div class="inst-title">Trabalho 3ª Etapa — Pesquisa Teórica e Execução Prática Individual</div>
  <table class="info-table">
    <tr>
      <td style="width: 50%;"><strong>Professor:</strong> Felipe Roncalli de Paula Carneiro</td>
      <td style="width: 50%; text-align: right;"><strong>Data de Entrega:</strong> 20 de Novembro de 2026</td>
    </tr>
    <tr>
      <td><strong>Aluno:</strong> Gabriel Possa</td>
      <td style="text-align: right;"><strong>Ambiente:</strong> Ollama Local &bull; Modelo Qwen2.5-Coder 7B (Q4_K_M)</td>
    </tr>
  </table>
</div>

{body_html}

<div class="footer-sig">
  Documento gerado para fins de avaliação acadêmica &bull; UNIPAC Barbacena &bull; Ciência da Computação
</div>

</body>
</html>
"""
    return full_html


def main() -> None:
    print(f"Lendo Markdown de: {MD_PATH}")
    html_content = build_html()
    HTML_PATH.write_text(html_content, encoding="utf-8")
    print(f"HTML intermediário salvo em: {HTML_PATH}")

    print("Convertendo HTML para PDF via LibreOffice...")
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        str(HTML_PATH),
        "--outdir",
        str(BASE_DIR),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Erro no LibreOffice: {res.stderr}")
        raise SystemExit(res.returncode)

    print(f"PDF gerado com sucesso em: {PDF_PATH}")


if __name__ == "__main__":
    main()
