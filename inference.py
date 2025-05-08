#!/usr/bin/env python
"""
Interactive distributed inference for DeepSeek-Coder.

Запускайте скрипт через torchrun / torch.distributed, чтобы
каждый процесс работал на своём GPU (у нас ― 1 GPU на узел).

Rank-0 (MASTER) показывает приглашение >>>, куда пользователь
вводит запрос (код или естественный язык). «Enter» на пустой
строке завершает ввод.  Команды  exit / quit  останавливают
работу всех процессов.

Остальные ранги получают тот же запрос через broadcast, делают
вычисления на своих GPU, но выводит результат только rank-0,
поэтому вычислительные ресурсы используются без конфликтов.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

# ───────────────────────── 1. инициализация Distributed ──────────────────────
dist.init_process_group(backend="nccl")
local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
rank = dist.get_rank()
world_size = dist.get_world_size()

# ─────────────────────────── 2. аргументы CLI ───────────────────────────────
parser = argparse.ArgumentParser(description="DeepSeek-Coder interactive REPL")
parser.add_argument(
    "--model_path",
    default="/opt/models/deepseek-coder-6.7b-base",
    help="Путь к модели или её repo-id на HuggingFace",
)
parser.add_argument("--max_new_tokens", type=int, default=256,
                    help="Сколько новых токенов генерировать")
parser.add_argument("--temperature", type=float, default=0.2,
                    help="Температура семплирования (0 = детерминизм)")
args = parser.parse_args()

# ─────────────────────── 3. загрузка токенизатора и модели ───────────────────
if rank == 0:
    print(f"[RANK 0] Загружаю модель «{args.model_path}» …", flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map={"": local_rank},        # один GPU на процесс
    trust_remote_code=True,
).eval()

def bcast(obj, src=0):
    "Утилита для broadcast произвольных python-объектов"
    buf = [obj]
    dist.broadcast_object_list(buf, src)
    return buf[0]

# ─────────────────────────── 4. REPL-цикл ─────────────────────────────────────
if rank == 0:
    print("\nDeepSeek-Coder готов. Введите запрос (код/инструкцию)."
          "\nПустая строка завершит ввод.  exit / quit — выход.\n")

while True:
    # -------- rank-0 читает ввод пользователя --------------------------------
    if rank == 0:
        user_lines = []
        sys.stdout.write(">>> ")
        sys.stdout.flush()
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line == "":
                break
            user_lines.append(line)
            sys.stdout.write("... ")
            sys.stdout.flush()
        prompt = "\n".join(user_lines).strip()
        if prompt.lower() in {"exit", "quit"} or prompt == "":
            prompt = None
    else:
        prompt = None

    # -------- рассылка запроса всем процессам --------------------------------
    prompt = bcast(prompt, src=0)
    if prompt is None:
        if rank == 0:
            print("[RANK 0] Завершаем работу …")
        break

    # -------- генерация ответа ------------------------------------------------
    inputs = tokenizer(prompt, return_tensors="pt").to(local_rank)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )

    # -------- вывод (только rank-0) ------------------------------------------
    if rank == 0:
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        completion = full_text[len(prompt):].lstrip("\n")
        print("\n=== 🧠 DeepSeek-Coder ответ ===\n" + completion + "\n")
