import requests
from bs4 import BeautifulSoup

base_url = "https://oset.nosu.ru/category/news/page/{}/"

for page_num in range(1, 100):
    url = base_url.format(page_num)
    response = requests.get(url)
    if response.status_code != 200: 
        break
    soup = BeautifulSoup(response.content, 'html.parser')

    article_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        print(href)
        if '/2026/' in href or '/2025/' in href or '/2024/' in href:
            article_links.append(href)
    print(article_links)
    for i, link in enumerate(article_links[:10]):
        try:
            if not link.startswith('http'):
                link = 'https://oset.nosu.ru' + link
            article_response = requests.get(link)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            title = article_soup.find('h1').get_text(strip=True)
            content_div = article_soup.find('div', class_='content-block content-text') 
            if content_div:
                paragraphs = content_div.find_all('p')
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                article_tag = article_soup.find('article')
                if article_tag:
                    text = article_tag.get_text(separator=' ', strip=True)
                else:
                    text = "Текст статьи не найден"
            with open('ossetian_news_corpus.txt', 'a', encoding='utf-8') as f:
                f.write(f"{title}. ")
                f.write(f"{text}\n")
                f.write("<|endoftext|>\n\n")
            print(f"Сохранена статья {i+1}: {title}")
        except Exception as e:
            print(f"Ошибка при загрузке статьи {link}: {e}")