import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _relax_numeric_validation():
    """Allow int values in `float` config fields (process-wide).

    `ai-sage/GigaChat3-10B-A1.8B` ships a config.json where float fields such
    as `routed_scaling_factor` are JSON integers. huggingface_hub 2.x validates
    config fields strictly (int is not a float) and aborts loading.

    The relaxation is applied to the whole process (every model loaded here).
    It relies on the private hook `huggingface_hub.dataclasses._validate_simple_type`;
    if that hook is absent, strict validation is kept unchanged instead of failing.
    """
    import huggingface_hub.dataclasses as hf_dataclasses

    original = getattr(hf_dataclasses, "_validate_simple_type", None)
    if original is None:
        return

    def tolerant(name, value, expected_type):
        if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
            return
        return original(name, value, expected_type)

    hf_dataclasses._validate_simple_type = tolerant


_relax_numeric_validation()

PUNCT = ".,;:!?…—–-\"'()[]{}«»"


def _next_word_distribution(model, tokenizer, prompt_text, top_k=200, max_new_tokens=8):
    """Суммирует вероятности по всем токенизациям слова (агрегация по словам).

    Разворачивает каждую из топ-`top_k` первых токенов до конца слова и
    складывает вероятности путей, дающих одинаковое слово. Так « ёл» + «очка»
    и другие варианты дают одну строку `ёлочка`, а не разные токены.
    """
    device = model.device
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
    logprobs = torch.log_softmax(logits, dim=-1)
    special_ids = [i for i in tokenizer.all_special_ids if 0 <= i < logprobs.shape[-1]]
    if special_ids:
        logprobs[torch.tensor(special_ids, device=device)] = float("-inf")

    top_k = min(top_k, logprobs.shape[-1])
    first_lp, first_ids = torch.topk(logprobs, top_k)

    batch = enc.input_ids.repeat(top_k, 1)
    batch = torch.cat([batch, first_ids.unsqueeze(1)], dim=1)
    attention_mask = torch.ones_like(batch)

    with torch.no_grad():
        generated = model.generate(
            batch,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    gen_ids = generated.sequences[:, prompt_len + 1:]
    step_lp = []
    for i, scores in enumerate(generated.scores):
        step_lp.append(torch.log_softmax(scores, dim=-1).gather(1, gen_ids[:, i:i + 1]).squeeze(1))

    agg = {}
    for k in range(top_k):
        pieces = [tokenizer.decode([first_ids[k].item()])]
        total_lp = first_lp[k].item()
        for i in range(gen_ids.shape[1]):
            token_id = gen_ids[k, i].item()
            if token_id == tokenizer.eos_token_id:
                break
            piece = tokenizer.decode([token_id])
            if piece and piece[0].isspace():
                break
            if piece and all(ch in PUNCT for ch in piece.strip()):
                break
            pieces.append(piece)
            total_lp += step_lp[i][k].item()

        word = "".join(pieces).strip().strip(PUNCT)
        if not any(ch.isalnum() for ch in word):
            continue
        agg[word] = agg.get(word, 0.0) + float(torch.exp(torch.tensor(total_lp)))

    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)


def _complete_words(model, tokenizer, prompt_text, token_ids, max_new_tokens=8):
    """Достраивает каждое слово до конца (жадно), чтобы показать слово, а не токен.

    Продолжение считается для ТЕКСТА `prompt_text + токен` (а не для отдельного id):
    промпт с добавленным токеном заново токенизируется и достраивается жадной
    генерацией. Отдельный токен вроде ' ко' — лишь начало слова; целое слово
    (`кошка`) получается только после нескольких шагов генерации.
    """
    words = []
    for token_id in token_ids.tolist():
        token_text = tokenizer.decode([token_id])
        inputs = tokenizer(prompt_text + token_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        text = (token_text + completion).strip()
        parts = text.split()
        word = parts[0] if parts else ""
        words.append(word.strip(".,;:!?—–-\"'()[]»«"))
    return words


def analyze_next_tokens(text, model_name, device="cpu"):
    """
    Анализирует текст и выводит топ-20 наиболее вероятных продолжений (токенов).
    
    Args:
        text (str): Входной текст (промпт).
        model_name (str): Идентификатор модели на HuggingFace.
        device (str): 'cpu' или 'cuda' (для GPU).
    """
    print(f"\n{'='*60}")
    print(f" АНАЛИЗ МОДЕЛИ: {model_name}")
    print(f"{'='*60}")
    print(f"📝 Входной текст: '{text}'")
    print("-"*60)

    try:
        # 1. Загрузка токенизатора и модели
        # trust_remote_code=True нужен для некоторых моделей (включая GigaChat и Qwen)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # float16 имеет смысл только на GPU; на CPU он медленный и часто нестабильный
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model.to(device)
            
        model.eval()

        # 2. Токенизация входного текста
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 3. Получение логитов (сырых оценок) для последнего токена
        with torch.no_grad():
            outputs = model(**inputs)
            # logits имеет форму [batch_size, sequence_length, vocab_size]
            # Нас интересует последний токен последовательности: [0, -1, :]
            last_token_logits = outputs.logits[0, -1, :]

        # 4. Преобразование логитов в вероятности через Softmax
        probabilities = torch.softmax(last_token_logits, dim=-1)

        # 5. Скрываем служебные токены (<|im_end|>, <|endoftext|> и т.п.),
        #    чтобы в таблице были слова, а не управляющие маркеры
        special_ids = [i for i in tokenizer.all_special_ids if 0 <= i < probabilities.shape[-1]]
        if special_ids:
            probabilities[torch.tensor(special_ids, device=probabilities.device)] = 0.0

        # 6. Поиск топ-20 наиболее вероятных токенов
        top_k = 20
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # 7. Достраиваем слова, чтобы показать не слог, а целое слово-продолжение
        words = _complete_words(model, tokenizer, text, top_indices)

        # 8. Вывод результатов
        print(f"{'Ранг':<6} {'ID':<10} {'Токен':<25} {'Слово':<18} {'Вероятность':<10}")
        print("-"*75)

        for i in range(top_k):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            # Декодируем ID обратно в текст
            token_text = tokenizer.decode([token_id])
            # Очищаем отображение (repr покажет спецсимволы типа \n, пробелы)
            display_text = repr(token_text)

            print(f"{i+1:<6} {token_id:<10} {display_text:<25} {words[i]:<18} {prob:.5f}")

        # 9. Агрегация по словам: сумма вероятностей по всем токенизациям слова
        word_dist = _next_word_distribution(model, tokenizer, text)
        print()
        print("-- агрегация по словам --")
        print(f"{'Ранг':<6} {'Слово':<18} {'Вероятность':<10}")
        print("-" * 36)
        for i, (word, prob) in enumerate(word_dist[:20], 1):
            print(f"{i:<6} {word:<18} {prob:<10.5f}")

    except Exception as e:
        print(f"❌ Ошибка при обработке модели {model_name}: {e}")
        import traceback
        traceback.print_exc()

# === НАСТРОЙКИ ===
# Текст для анализа (ваш пример с лекции)
INPUT_TEXT = "В лесу родилась рыжая"
#INPUT_TEXT = "В лесу родилась"

# Список моделей для проверки.
# Важно: для анализа «продолжения текста» нужны pretrain/base-модели (без -Instruct/-Chat),
# иначе модель обучена отвечать в диалоге и подмешивает комментарии и служебные токены.
MODELS_TO_CHECK = [
#    "ai-forever/rugpt3medium_based_on_gpt2",  # русская base-модель: знает «В лесу родилась ёлочка»
#    "Qwen/Qwen2.5-1.5B",                      # Qwen base (мультиязычная, без instruct)
    "ai-sage/GigaChat3-10B-A1.8B-base"        # GigaChat base (без instruct)
]

if __name__ == "__main__":
    # Автоматическое определение устройства (GPU если есть, иначе CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Используемое устройство: {device.upper()}")
    
    for model_name in MODELS_TO_CHECK:
        analyze_next_tokens(INPUT_TEXT, model_name, device=device)