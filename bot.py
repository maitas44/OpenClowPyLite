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

import time
last_activity_time = time.time()
needs_improvement = False
last_chat_id = None

def load_whitelist():
    try:
        with open("whitelist.txt", "r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def is_authorized(chat_id):
    whitelist = load_whitelist()
    if not whitelist: # If empty/missing, allow all for now but warn
        return True
    return str(chat_id) in whitelist

async def check_inactivity(context: ContextTypes.DEFAULT_TYPE):
    global last_activity_time, needs_improvement, last_chat_id
    
    # Check if 3600 seconds (1 hour) have passed since the last activity AND we need improvement
    if needs_improvement and (time.time() - last_activity_time > 3600):
        needs_improvement = False # Only do this once per idle session
        
        logging.info("1 hour of inactivity detected. Asking Gemini to improve prompts...")
        new_prompt_text = await agent.improve_prompt()
        
        if new_prompt_text and last_chat_id:
            logging.info("Prompt successfully improved and saved!")
            
            # Send the new prompt to the user (truncate to Telegram's 4096 char limit if necessary)
            safe_text = new_prompt_text[:4000] if len(new_prompt_text) > 4000 else new_prompt_text
            message = f"🧠 **I used my idle time to improve my instructions!**\n\nHere is my new internal prompt:\n\n```text\n{safe_text}\n```"
            
            try:
                await context.bot.send_message(chat_id=last_chat_id, text=message, parse_mode='Markdown')
            except Exception as e:
                logging.error(f"Failed to send improved prompt to chat: {e}")
                
        else:
            logging.error("Failed to improve prompt.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_authorized(chat_id):
        await context.bot.send_message(chat_id=chat_id, text="Unauthorized access. Please contact the administrator.")
        return

    global last_chat_id
    last_chat_id = chat_id
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
    chat_id = update.effective_chat.id
    if not is_authorized(chat_id):
        await context.bot.send_message(chat_id=chat_id, text="Unauthorized access.")
        return

    global last_activity_time, needs_improvement, last_chat_id
    last_activity_time = time.time()
    needs_improvement = True
    last_chat_id = chat_id
    
    url = ' '.join(context.args)
    if not url:
        await context.bot.send_message(chat_id=chat_id, text="Please provide a URL. Usage: /browse <url>")
        return

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Navigating to {url}...")
    
    msg = await browser.navigate(url)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    
    # Take initial screenshot and show it
    screenshot = await browser.take_screenshot()
    if screenshot:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=screenshot)
    
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_activity_time, needs_improvement, last_chat_id
    last_activity_time = time.time()
    needs_improvement = True
    last_chat_id = update.effective_chat.id
    
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if not user_text:
        return

    # Check for image generation request
    if user_text.lower().startswith("generate image"):
        await context.bot.send_message(chat_id=chat_id, text="Generating image...")
        result = await agent.generate_image(user_text)
        
        if result.startswith("IMAGE:"):
            image_path = result.split(":", 1)[1]
            try:
                with open(image_path, 'rb') as photo:
                    await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="Here is your generated image!")
            except Exception as e:
                await context.bot.send_message(chat_id=chat_id, text=f"Error sending image: {e}")
        else:
            await context.bot.send_message(chat_id=chat_id, text=result)
        return

async def _solve_autonomous(chat_id: int, user_text: str, context: ContextTypes.DEFAULT_TYPE):
    """
    Runs an autonomous loop, asking the agent for actions until it signals it is done or 10 minutes pass.
    """
    from telegram.constants import ChatAction
    start_time = time.time()
    max_duration = 600  # 10 minutes
    
    await context.bot.send_message(chat_id=chat_id, text=f"Starting autonomous task: '{user_text}'. (Timeout: 10m)")
    
    while time.time() - start_time < max_duration:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # 1. Take current screenshot
        screenshot = await browser.take_screenshot()
        
        if not screenshot:
            await context.bot.send_message(chat_id=chat_id, text="Browser was not active. Auto-starting and navigating to DuckDuckGo Lite...")
            await browser.navigate("https://lite.duckduckgo.com/lite/")
            screenshot = await browser.take_screenshot()
            
            if not screenshot:
                await context.bot.send_message(chat_id=chat_id, text="Failed to take screenshot even after auto-starting the browser. Task aborted.")
                return

        # 2. Ask Agent (Gemini) what to do
        response_text, is_done, tts_path = await agent.analyze_and_act(user_text, screenshot, chat_id)
        
        # Send intermediate progress updates to the user (truncate large reads to avoid spam)
        short_response = response_text[:200] + "..." if len(response_text) > 200 else response_text
        if not is_done:
             await context.bot.send_message(chat_id=chat_id, text=f"(Working) {short_response}")
             
        else:
             # Task completed or answered
             await context.bot.send_message(chat_id=chat_id, text=f"✅ Task Completed:\n{response_text}")
             
             # If TTS was generated, send it
             if tts_path and os.path.exists(tts_path):
                 try:
                     with open(tts_path, 'rb') as voice:
                         await context.bot.send_voice(chat_id=chat_id, voice=voice)
                     os.remove(tts_path) # Cleanup
                 except Exception as e:
                     logging.error(f"Error sending TTS record: {e}")

             final_screenshot = await browser.take_screenshot()
             if final_screenshot:
                 await context.bot.send_photo(chat_id=chat_id, photo=final_screenshot)
             return
             
    # If we get here, the loop timed out
    await context.bot.send_message(chat_id=chat_id, text="⏱️ Task timed out after 10 minutes. I was unable to fulfill the request.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_authorized(chat_id):
        await context.bot.send_message(chat_id=chat_id, text="Unauthorized access.")
        return

    global last_activity_time, needs_improvement, last_chat_id
    last_activity_time = time.time()
    needs_improvement = True
    last_chat_id = chat_id
    
    user_text = update.message.text

    if not user_text:
        return

    # Hand off to the autonomous loop
    asyncio.create_task(_solve_autonomous(chat_id, user_text, context))

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_authorized(chat_id):
        await context.bot.send_message(chat_id=chat_id, text="Unauthorized access.")
        return

    global last_activity_time, needs_improvement, last_chat_id
    last_activity_time = time.time()
    needs_improvement = True
    last_chat_id = chat_id
    
    message = update.message
    
    if message.voice:
        audio_file = await message.voice.get_file()
    elif message.audio:
        audio_file = await message.audio.get_file()
    else:
        return

    await context.bot.send_message(chat_id=chat_id, text="🎙️ Listening and transcribing...")
    
    temp_path = f"temp_audio_{chat_id}.ogg"
    await audio_file.download_to_drive(temp_path)
    
    transcript = await agent.transcribe_audio(temp_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    if transcript.startswith("Error"):
        await context.bot.send_message(chat_id=chat_id, text=transcript)
        return
        
    await context.bot.send_message(chat_id=chat_id, text=f"🗣️ I heard: '{transcript}'")
    
    # Hand off to autonomous logic
    asyncio.create_task(_solve_autonomous(chat_id, transcript, context))

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
    browse_handler = CommandHandler(['browse', 'browser'], browse_command)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    audio_handler = MessageHandler(filters.VOICE | filters.AUDIO, handle_audio)

    application.add_handler(start_handler)
    application.add_handler(browse_handler)
    application.add_handler(message_handler)
    application.add_handler(audio_handler)

    # Start the background job to monitor inactivity every 5 seconds
    application.job_queue.run_repeating(check_inactivity, interval=5)

    print("Bot is polling...")
    application.run_polling()
