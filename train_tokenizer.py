import json
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

corpus_file = 'ossetian_corpus.jsonl'

def get_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                yield data["text"]
            except json.JSONDecodeError:
                print("Ошибка чтения строки, пропускаем...")
                continue
tokenizer = Tokenizer(BPE(byte_fallback=True))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=10000,          
    min_frequency=2,           
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"] 
)
print("Начинаем обучение токенизатора...")
texts_iterator = get_texts(corpus_file)
tokenizer.train_from_iterator(texts_iterator, trainer=trainer)
print("Обучение завершено!")

output_dir = "ossetian_bpe_tokenizer"
os.makedirs(output_dir, exist_ok=True)
tokenizer.save(f"{output_dir}/tokenizer.json")
print(f"Токенизатор сохранен в папку: {output_dir}")

test_text = "Цæгат Ирыстоны Хетæгкаты Къостайы номыл паддзахадон университет."
encoded = tokenizer.encode(test_text)
print(f"\nТестовый текст: {test_text}")
print(f"ID токенов: {encoded.ids[:20]}...")
print(f"Декодированный текст: {tokenizer.decode(encoded.ids)}")