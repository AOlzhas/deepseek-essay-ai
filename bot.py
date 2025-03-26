from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import pipeline

TOKEN = "ВАШ_TELEGRAM_BOT_TOKEN"
model = pipeline("text-generation", model="gpt2")

async def start(update: Update, context):
    await update.message.reply_text("Привет! Я бот для генерации текста.")

async def generate_text(update: Update, context):
    prompt = update.message.text
    response = model(prompt, max_length=100)[0]['generated_text']
    await update.message.reply_text(response)

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_text))

print("Бот запущен...")
app.run_polling()