import logging
import telebot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes
)
from openai import OpenAI
import matplotlib.pyplot as plt
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List

# Настройки логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_TOKEN = '7530307474:AAEe23XCzPa3x_Cn6HrfS6dNir9uJQo4Ixo'
OPENAI_API_KEY = 'sk-eojihWMYuwlwO4oNjNMX8DbkkkBtLg7I'

# Инициализация OpenAI
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.proxyapi.ru/openai/v1",
)

# Состояния диалога
CALCULATING, AI_CHAT, QUIZ = range(3)

# Инициализация БД
conn = sqlite3.connect('fuel_bot.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users
                (user_id INTEGER PRIMARY KEY,
                 history TEXT,
                 language TEXT DEFAULT 'ru',
                 score INTEGER DEFAULT 0)''')
conn.commit()

# Локализация
TRANSLATIONS = {
    'ru': {
        'welcome': 'Привет, {name}! Я экспертный бот по сжиганию топлива.',
        'main_menu': 'Выберите действие:',
        'invalid_topic': 'Я могу отвечать только на вопросы по сжиганию топлива.',
        'calculation_prompt': 'Введите количество полезного топлива (Дж):',
        'spent_prompt': 'Введите затраченное топливо (Дж):',
        'result': 'Коэффициент сжигания: {result:.2f}%',
        'history_empty': 'История расчетов пуста.',
        'leaderboard': '🏆 Топ пользователей:\n{leaderboard}',
        'error': '⚠️ Ошибка обработки запроса',
        'ai_prompt': 'Задайте ваш вопрос по сжиганию топлива:',
        'back_button': '🔙 Назад'
    },
    'en': {
        'welcome': 'Hello, {name}! I am a fuel combustion expert bot.',
        'main_menu': 'Choose action:',
        'invalid_topic': 'I can only answer questions about fuel combustion.',
        'calculation_prompt': 'Enter useful fuel amount (J):',
        'spent_prompt': 'Enter spent fuel amount (J):',
        'result': 'Combustion coefficient: {result:.2f}%',
        'history_empty': 'Calculation history is empty.',
        'leaderboard': '🏆 Leaderboard:\n{leaderboard}',
        'error': '⚠️ Request processing error',
        'ai_prompt': 'Ask your fuel combustion question:',
        'back_button': '🔙 Back'
    }
}


class UserManager:
    @staticmethod
    def get_user(user_id: int) -> Dict:
        cursor.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        user = cursor.fetchone()
        if not user:
            cursor.execute("INSERT INTO users (user_id, history) VALUES (?, ?)",
                           (user_id, '[]'))
            conn.commit()
            return {'user_id': user_id, 'history': [], 'language': 'ru', 'score': 0}
        return {
            'user_id': user[0],
            'history': json.loads(user[1]),
            'language': user[2],
            'score': user[3]
        }

    @staticmethod
    def update_user(user_id: int, **kwargs):
        user = UserManager.get_user(user_id)
        for key, value in kwargs.items():
            user[key] = value
        cursor.execute("UPDATE users SET history=?, language=?, score=? WHERE user_id=?",
                       (json.dumps(user['history']), user['language'], user['score'], user_id))
        conn.commit()


class EcoQuiz:
    QUESTIONS = {
        'ru': [
            {
                'question': 'Оптимальный коэффициент сжигания?',
                'options': ['50-60%', '70-80%', '90-95%'],
                'correct': 2
            }
        ],
        'en': [
            {
                'question': 'Optimal combustion coefficient?',
                'options': ['50-60%', '70-80%', '90-95%'],
                'correct': 2
            }
        ]
    }

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.current = 0
        self.score = 0
        self.lang = UserManager.get_user(user_id)['language']

    @property
    def next_question(self):
        if self.current < len(self.QUESTIONS[self.lang]):
            return self.QUESTIONS[self.lang][self.current]
        return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    UserManager.get_user(user.id)
    lang = UserManager.get_user(user.id)['language']

    keyboard = [
        [InlineKeyboardButton("🧮 Калькулятор" if lang == 'ru' else "🧮 Calculator", callback_data='calc'),
         InlineKeyboardButton("🤖 ИИ-эксперт" if lang == 'ru' else "🤖 AI Expert", callback_data='ai')],
        [InlineKeyboardButton("📈 История" if lang == 'ru' else "📈 History", callback_data='history'),
         InlineKeyboardButton("🏆 Топ" if lang == 'ru' else "🏆 Top", callback_data='leaderboard')],
        [InlineKeyboardButton("❓ Викторина" if lang == 'ru' else "❓ Quiz", callback_data='quiz')]
    ]

    await update.message.reply_text(
        TRANSLATIONS[lang]['welcome'].format(name=user.first_name),
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    handlers = {
        'calc': start_calculation,
        'ai': start_ai_chat,
        'history': show_history,
        'leaderboard': show_leaderboard,
        'quiz': start_quiz,
        'back': start  # Обработка кнопки "Назад"
    }
    if query.data in handlers:
        return await handlers[query.data](update, context)
    return ConversationHandler.END


async def start_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = UserManager.get_user(update.effective_user.id)['language']
    await update.callback_query.message.reply_text(TRANSLATIONS[lang]['calculation_prompt'])
    return CALCULATING


async def process_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user = UserManager.get_user(user_id)
    lang = user['language']

    try:
        value = float(update.message.text)
    except ValueError:
        await update.message.reply_text(f"⚠️ {TRANSLATIONS[lang]['error']} - Invalid number")
        return CALCULATING

    if 'calc_data' not in context.user_data:
        context.user_data['calc_data'] = {'useful': value}
        await update.message.reply_text(TRANSLATIONS[lang]['spent_prompt'])
        return CALCULATING

    context.user_data['calc_data']['spent'] = value
    useful = context.user_data['calc_data']['useful']
    spent = context.user_data['calc_data']['spent']

    coefficient = (useful / spent) * 100 if spent != 0 else 0
    response = TRANSLATIONS[lang]['result'].format(result=coefficient)

    history = user['history']
    history.append({
        'date': datetime.now().isoformat(),
        'type': 'calculation',
        'result': coefficient
    })
    UserManager.update_user(user_id, history=history, score=user['score'] + 10)

    await update.message.reply_text(response)
    context.user_data.clear()
    return ConversationHandler.END


async def start_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = UserManager.get_user(update.effective_user.id)['language']
    keyboard = [
        [InlineKeyboardButton(TRANSLATIONS[lang]['back_button'], callback_data='back')]
    ]
    await update.callback_query.message.reply_text(
        TRANSLATIONS[lang]['ai_prompt'],
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return AI_CHAT


AI_SYSTEM_PROMPT = {
    'ru': "Ты эксперт по сжиганию топлива. Отвечай только на вопросы по: горению, топливным системам, энергоэффективности.",
    'en': "You are a fuel combustion expert. Only answer questions about: combustion, fuel systems, energy efficiency."
}


async def process_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user = UserManager.get_user(user_id)
    question = update.message.text.lower()
    lang = user['language']

    keywords = ['топлив', 'горен', 'энерг'] if lang == 'ru' else ['fuel', 'combust', 'energy']
    if not any(kw in question for kw in keywords):
        await update.message.reply_text(TRANSLATIONS[lang]['invalid_topic'])
        return AI_CHAT

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT[lang]},
                {"role": "user", "content": question}
            ],
            temperature=0.4,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

        history = user['history']
        history.append({
            'date': datetime.now().isoformat(),
            'type': 'ai_chat',
            'question': question,
            'answer': answer
        })
        UserManager.update_user(user_id, history=history, score=user['score'] + 5)

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        await update.message.reply_text(TRANSLATIONS[lang]['error'])

    # Возвращаемся в главное меню
    await start(update, context)
    return ConversationHandler.END


async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = UserManager.get_user(update.effective_user.id)
    history = user['history']
    lang = user['language']

    if not history:
        await update.callback_query.message.reply_text(TRANSLATIONS[lang]['history_empty'])
        return

    dates = [entry['date'][:10] for entry in history if entry['type'] == 'calculation']
    values = [entry['result'] for entry in history if entry['type'] == 'calculation']

    if not values:
        await update.callback_query.message.reply_text(TRANSLATIONS[lang]['history_empty'])
        return

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker='o')
    plt.title('История расчетов' if lang == 'ru' else 'Calculation History')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{user["user_id"]}_plot.png')

    with open(f'{user["user_id"]}_plot.png', 'rb') as photo:
        await update.callback_query.message.reply_photo(photo=photo)

    os.remove(f'{user["user_id"]}_plot.png')


async def show_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cursor.execute("SELECT user_id, score FROM users ORDER BY score DESC LIMIT 10")
    top_users = cursor.fetchall()
    lang = UserManager.get_user(update.effective_user.id)['language']

    leaderboard = "\n".join(
        f"{i + 1}. User {user[0]}: {user[1]} pts" for i, user in enumerate(top_users)
    )
    await update.callback_query.message.reply_text(
        TRANSLATIONS[lang]['leaderboard'].format(leaderboard=leaderboard)
    )


async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    context.user_data['quiz'] = EcoQuiz(user_id)
    return await ask_quiz_question(update, context)


async def ask_quiz_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    quiz = context.user_data['quiz']
    question = quiz.next_question
    lang = quiz.lang

    if not question:
        UserManager.update_user(
            quiz.user_id,
            score=UserManager.get_user(quiz.user_id)['score'] + quiz.score
        )
        await update.callback_query.message.reply_text(
            f"Викторина завершена! Счет: {quiz.score}" if lang == 'ru'
            else f"Quiz complete! Score: {quiz.score}"
        )
        context.user_data.clear()
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton(opt, callback_data=str(i))]
        for i, opt in enumerate(question['options'])
    ]

    await update.callback_query.message.reply_text(
        question['question'],
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return QUIZ


async def process_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    quiz = context.user_data['quiz']
    lang = quiz.lang

    if int(query.data) == quiz.next_question['correct']:
        quiz.score += 20
        await query.answer("Правильно!" if lang == 'ru' else "Correct!")
    else:
        await query.answer("Неправильно!" if lang == 'ru' else "Wrong!")

    quiz.current += 1
    return await ask_quiz_question(update, context)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Error:", exc_info=context.error)
    if update.message:
        lang = UserManager.get_user(update.message.from_user.id)['language']
        await update.message.reply_text(TRANSLATIONS[lang]['error'])


def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start), CallbackQueryHandler(button_handler)],
        states={
            CALCULATING: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_calculation)],
            AI_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_ai_chat)],
            QUIZ: [CallbackQueryHandler(process_quiz_answer)]
        },
        fallbacks=[CommandHandler('cancel', lambda u, c: ConversationHandler.END)]
    )

    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
