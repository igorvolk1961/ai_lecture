"""Предварительный расчёт вероятностей следующего токена/слова через llama.cpp (GGUF).

Нужен запущенный `llama-server` с моделью. Пример запуска:

    llama-server -m GigaChat3-10B-A1.8B-base-Q4_K_M.gguf --port 8080 -c 2048

Считаются две таблицы:
  * по токенам — распределение следующего токена;
  * по словам — сумма вероятностей по всем токенизациям слова (агрегация),
    что убирает артефакт «одно слово разбито на разные токены».

Результат пишется в текстовый файл, чтобы показывать его на лекции без запуска модели.
"""

import json
import math
import urllib.request

SERVER_URL = "http://127.0.0.1:8080"
PROMPTS = [
    "В лесу родилась",
    "В лесу родилась рыжая",
]
N_PROBS = 20
TOP_K_WORDS = 100
MAX_WORD_TOKENS = 8
OUT_PATH = "scripts/token_result_gigachat.txt"

PUNCT = ".,;:!?…—–-\"'()[]{}«»"


def _request(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        SERVER_URL + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def top_probs(prompt, n_probs):
    result = _request(
        {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": n_probs,
            "temperature": 0,
            "cache_prompt": False,
            "post_sampling_probs": False,
        },
    )
    entries = result["completion_probabilities"][0]["top_logprobs"]
    return [(e["token"], e["logprob"]) for e in entries]


def next_word_distribution(prompt, top_k=TOP_K_WORDS, max_new_tokens=MAX_WORD_TOKENS):
    """Агрегация по словам: разворачиваем каждую топ-ветку до границы слова."""
    agg = {}
    for token_text, logprob in top_probs(prompt, top_k):
        result = _request(
            {
                "prompt": prompt + token_text,
                "n_predict": max_new_tokens,
                "n_probs": 1,
                "temperature": 0,
                "cache_prompt": False,
            },
        )
        pieces = [token_text]
        for entry in result.get("completion_probabilities", []):
            piece = entry["token"]
            if piece and piece[0].isspace():
                break
            if piece and all(ch in PUNCT for ch in piece.strip()):
                break
            pieces.append(piece)
            logprob += entry["logprob"]

        word = "".join(pieces).strip().strip(PUNCT)
        if not any(ch.isalnum() for ch in word):
            continue
        agg[word] = agg.get(word, 0.0) + math.exp(logprob)

    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)


def _token_table(prompt):
    lines = [f"{'Ранг':<6}{'Токен':<18}{'Вероятность':<12}", "-" * 36]
    for rank, (token, logprob) in enumerate(top_probs(prompt, N_PROBS), 1):
        lines.append(f"{rank:<6}{token!r:<18}{math.exp(logprob):<12.5f}")
    return lines


def _word_table(prompt):
    lines = [f"{'Ранг':<6}{'Слово':<18}{'Вероятность':<12}", "-" * 36]
    for rank, (word, prob) in enumerate(next_word_distribution(prompt)[:20], 1):
        lines.append(f"{rank:<6}{word:<18}{prob:<12.5f}")
    return lines


def main():
    blocks = []
    for prompt in PROMPTS:
        blocks.append(f"Вход: {prompt!r}\n")
        blocks.append("-- по токенам --")
        blocks.extend(_token_table(prompt))
        blocks.append("")
        blocks.append("-- агрегация по словам --")
        blocks.extend(_word_table(prompt))
        blocks.append("")
        blocks.append("=" * 60)
        blocks.append("")

    text = "\n".join(blocks)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
