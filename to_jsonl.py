import json
import re

def convert_txt_to_jsonl(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file_path} не найден.")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    articles = re.split(r'\s*<\|endoftext\|>\s*', raw_text)
    with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
        for i, article in enumerate(articles):
            clean_article = article.strip()
            if clean_article:
                json_object = {"text": clean_article}
                jsonl_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
            else:
                print(f"Предупреждение: Статья {i+1} пуста и была пропущена.")

    print(f"Готово! Файл сохранен как: {output_file_path}")

input_file = 'ossetian_news_corpus.txt' 
output_file = 'ossetian_corpus.jsonl' 

convert_txt_to_jsonl(input_file, output_file)