# AI Critical Thinking Project

## Структура проекта
```
ai_critical_thinking/
├── data/                  # Данные
│   └── dataset.xlsx       # Пример датасета
├── app.py                 # Gradio-интерфейс
├── bot.py                 # Telegram-бот
└── README.md              # Этот файл
```

## Установка
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск
- Gradio: `python app.py`
- Бот: `python bot.py`