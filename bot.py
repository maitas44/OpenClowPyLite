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
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text="Hello! I am your OpenClawPyLite bot.\n\n"
             "I am spinning up the browser and navigating to http://neverssl.com/..."
    )
    
    # Auto-navigate on start
    await browser.navigate("http://neverssl.com/")
    screenshot = await browser.take_screenshot()
    
    if screenshot:
        await context.bot.send_photo(chat_id=chat_id, photo=screenshot)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Browser is ready! Tell me what to search for or click next."
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
    from telegram.constants import ChatAction
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # 1. Take current screenshot
    screenshot = await browser.take_screenshot()
    
    if not screenshot:
        # Browser might not be started or previous navigation failed
        await context.bot.send_message(chat_id=chat_id, text="Browser was not active. Auto-starting and navigating to DuckDuckGo...")
        await browser.navigate("https://duckduckgo.com/")
        screenshot = await browser.take_screenshot()
        
        if not screenshot:
            await context.bot.send_message(chat_id=chat_id, text="Failed to take screenshot even after auto-starting the browser. Please try /start.")
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
    try:
        with open("telegramapikey.txt", "r") as f:
            bot_token = f.read().strip()
    except FileNotFoundError:
        print("Error: telegramapikey.txt file not found. Please create it and add your Telegram Bot Token.")
        exit(1)
        
    if not bot_token:
        print("Error: telegramapikey.txt is empty.")
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
