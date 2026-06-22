import torch
import torch.nn as nn
from tokenizers import Tokenizer
import time
from train import GPTModel

def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=100, 
    context_size=128, 
    temperature=0.7,
    top_k=10,
    device="cpu"
):
    model.eval()
    model.to(device)
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(encoded).unsqueeze(0).to(device)
    generated_tokens = []
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                mask = torch.zeros_like(logits)
                mask.scatter_(1, top_k_indices, 1.0)
                logits = logits * mask
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == 0: 
                break
            input_ids = torch.cat((input_ids, next_token), dim=1)
            generated_tokens.append(next_token.item())
    full_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(full_ids)

def interactive_chat(model, tokenizer, config, device="cpu"):
    """Интерактивный режим - вводите промпты и получайте ответы"""
    print("\n" + "="*50)
    print("Интерактивный режим. Введите 'exit' для выхода.")
    print("="*50)
    
    while True:
        prompt = input("\nВведите промпт: ")
        if prompt.lower() in ['exit', 'quit', 'выход']:
            break
        
        print("\nГенерация...")
        start_time = time.time()
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            context_size=config["context_length"],
            temperature=0.7,
            top_k=10,
            device=device
        )
        
        end_time = time.time()
        print(f"\nСгенерировано за {end_time - start_time:.2f} сек:")
        print("-"*50)
        print(generated)
        print("-"*50)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    try:
        tokenizer = Tokenizer.from_file("ossetian_bpe_tokenizer/tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()
        print(f"Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Ошибка загрузки токенизатора: {e}")
        print("Убедитесь, что файл tokenizer.json находится в папке ossetian_bpe_tokenizer/")
        return
    
    config = {
        "vocab_size": vocab_size,
        "context_length": 128,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    print("Model config:", config)
    
    # 3. Загружаем модель
    print("Loading model...")
    model = GPTModel(config)
    
    try:
        model.load_state_dict(torch.load("ossetian_gpt_model.pth", map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Ошибка: Файл ossetian_gpt_model.pth не найден.")
        print("Сначала обучите модель с помощью train.py")
        return
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "="*50)
    print("Выберите режим:")
    print("1. Тестовые промпты (быстрое тестирование)")
    print("2. Интерактивный режим (диалог)")
    print("3. Сравнение промптов (генерация нескольких вариантов)")
    print("="*50)
    
    mode = input("Ваш выбор (1/2/3): ").strip()
    
    if mode == "1":
        test_prompts = [
            "ЦИПУ-йы",
            "Студенттæ",
            "Университеты",
            "Цæгат Ирыстоны",
            "Экономикæйы",
            "Хетæгкаты Къостайы номыл"
        ]
        
        print("\n" + "="*50)
        print("Тестирование на стандартных промптах:")
        print("="*50)
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            print(f"Промпт: {prompt}")
            print("="*50)
            
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
                context_size=config["context_length"],
                temperature=0.7,
                top_k=10,
                device=device
            )
            print(f"Результат: {generated}\n")
    
    elif mode == "2":
        interactive_chat(model, tokenizer, config, device)
    
    elif mode == "3":
        prompt = input("\nВведите промпт для сравнения: ")
        temperatures = [0.3, 0.7, 1.0, 1.5]
        
        print("\n" + "="*50)
        print(f"Сравнение генераций для промпта: '{prompt}'")
        print("="*50)
        
        for temp in temperatures:
            print(f"\n--- Температура: {temp} ---")
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=200,
                context_size=config["context_length"],
                temperature=temp,
                top_k=20,
                device=device
            )
            print(f"Результат: {generated}\n")
    
    else:
        print("Неверный выбор. Запустите программу снова.")

if __name__ == "__main__":
    main()