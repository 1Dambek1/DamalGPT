import json
from transformers import BertTokenizerFast

# Путь к JSON-файлу
file_path = r"./datas/arxivData.json"

# Чтение данных
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Создаём fast-токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
print(tokenizer.vocab_size)
# Добавляем специальный токен-разделитель
special_token = "[ARTICLE_END]"
if special_token not in tokenizer.get_vocab():
    tokenizer.add_tokens([special_token])

# Токенизация по статьям с добавлением разделителя
all_tokens = []
for article in data:
    summary = article.get("summary", "")
    # Токенизируем статью
    tokens = tokenizer.encode(summary, add_special_tokens=True)
    # Добавляем токены разделителя
    all_tokens.extend(tokens + tokenizer.encode(special_token, add_special_tokens=False))


