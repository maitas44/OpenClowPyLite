import os
import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from browser import BrowserManager
from agent import Agent

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize global instances
browser = BrowserManager()
agent = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello! I am your OpenClaw-inspired bot.\n\n"
             "I can browse the web for you using Gemini 3 Flash.\n"
             "Use /browse <url> to start, or just chat with me!"
    )

async def browse_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = ' '.join(context.args)
    if not url:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please provide a URL. Usage: /browse <url>")
        return

    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Navigating to {url}...")
    
    msg = await browser.navigate(url)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    
    # Take initial screenshot and show it
    screenshot = await browser.take_screenshot()
    if screenshot:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=screenshot)
    
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if not user_text:
        return

    # Check for image generation request
    if user_text.lower().startswith("generate image"):
        await context.bot.send_message(chat_id=chat_id, text="Generating image...")
        result = await agent.generate_image(user_text)
        await context.bot.send_message(chat_id=chat_id, text=result)
        return

    # Standard browser interaction
    await context.bot.send_typing(chat_id=chat_id)

    # 1. Take current screenshot
    screenshot = await browser.take_screenshot()
    
    if not screenshot:
        # Browser might not be started
        await context.bot.send_message(chat_id=chat_id, text="Browser is not active. Use /browse <url> first.")
        return

    # 2. Ask Agent (Gemini) what to do
    response = await agent.analyze_and_act(user_text, screenshot)
    
    await context.bot.send_message(chat_id=chat_id, text=response)

    # 3. If action resulted in visual change, send new screenshot
    # (Simple heuristic: always send screenshot after action for now)
    new_screenshot = await browser.take_screenshot()
    if new_screenshot:
        await context.bot.send_photo(chat_id=chat_id, photo=new_screenshot)

if __name__ == '__main__':
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        exit(1)

    # Initialize Agent
    try:
        agent = Agent(browser)
    except ValueError as e:
        print(f"Error initializing Agent: {e}")
        exit(1)

    application = ApplicationBuilder().token(bot_token).build()
    
    start_handler = CommandHandler('start', start)
    browse_handler = CommandHandler('browse', browse_command)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)

    application.add_handler(start_handler)
    application.add_handler(browse_handler)
    application.add_handler(message_handler)

    print("Bot is polling...")
    application.run_polling()
