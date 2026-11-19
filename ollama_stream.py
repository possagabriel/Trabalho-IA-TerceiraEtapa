#!/usr/bin/env python3
"""Gera texto via Ollama local, exibe o streaming e salva a resposta."""

import argparse
import json
import math
import sys
from pathlib import Path

import requests


DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_PROMPT = "Escreva um algoritmo de ordenação Bubble Sort em C bem documentado."
EXPERIMENT_TEMPERATURES = (0.0, 0.9, 1.5)


def generate(args: argparse.Namespace, temperature: float, output: Path) -> bool:
    """Executa uma chamada streaming e grava somente o texto gerado."""
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "options": {
            "temperature": temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
        },
        "stream": True,
    }

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        finished = False
        with requests.post(
            args.url,
            json=payload,
            stream=True,
            timeout=(10, args.timeout),
        ) as response:
            response.raise_for_status()
            if not response.encoding:
                response.encoding = "utf-8"
            print(f"\n--- Modelo: {args.model} | temperatura: {temperature} ---\n")
            with output.open("w", encoding="utf-8") as file:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    fragment = data.get("response", "")
                    if fragment:
                        print(fragment, end="", flush=True)
                        file.write(fragment)
                        file.flush()
                    if data.get("done", False):
                        finished = True
                        print("\n\n--- Geração concluída ---")
                        print(f"Tokens avaliados: {data.get('eval_count', 0)}")
                        if data.get("done_reason"):
                            print(f"Motivo do término: {data['done_reason']}")
            if not finished:
                raise requests.exceptions.RequestException(
                    "O fluxo terminou sem sinalizar a conclusão da geração."
                )
            print(f"Resposta salva em: {output}")
        return True
    except requests.exceptions.ConnectionError:
        print(
            "Erro: não foi possível conectar ao Ollama em localhost:11434. "
            "Inicie o serviço e confirme que o modelo está instalado.",
            file=sys.stderr,
        )
    except requests.exceptions.HTTPError as exc:
        print(f"Erro HTTP do Ollama: {exc}", file=sys.stderr)
    except requests.exceptions.Timeout:
        print(
            f"Erro: a geração excedeu o tempo limite de {args.timeout} segundos.",
            file=sys.stderr,
        )
    except (requests.exceptions.RequestException, json.JSONDecodeError, OSError) as exc:
        print(f"Erro durante a geração ou gravação: {exc}", file=sys.stderr)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consome o endpoint local do Ollama com streaming e salva a resposta."
    )
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Modelo já instalado no Ollama.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint HTTP de geração do Ollama.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt enviado ao modelo.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperatura da amostragem.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling (0 a 1).")
    parser.add_argument("--num-predict", type=int, default=600, help="Máximo de tokens gerados.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Tempo máximo de espera por resposta, em segundos (padrão: 600).",
    )
    parser.add_argument("--output", type=Path, default=Path("resposta_ollama.txt"), help="Arquivo de saída.")
    parser.add_argument(
        "--experimento",
        action="store_true",
        help="Executa o mesmo prompt nas temperaturas 0.0, 0.9 e 1.5.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.top_p) or not 0.0 <= args.top_p <= 1.0:
        parser.error("--top-p deve estar entre 0 e 1.")
    if args.num_predict < 1:
        parser.error("--num-predict deve ser maior que zero.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature não pode ser negativa.")
    if args.timeout < 1:
        parser.error("--timeout deve ser maior que zero.")
    return args


def main() -> int:
    args = parse_args()
    if args.experimento:
        success = True
        for temperature in EXPERIMENT_TEMPERATURES:
            output = args.output.parent / f"experimento_temp_{temperature:.1f}.txt"
            success = generate(args, temperature, output) and success
        return 0 if success else 1

    return 0 if generate(args, args.temperature, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
