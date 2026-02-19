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
    
async def _solve_autonomous(chat_id: int, user_text: str, context: ContextTypes.DEFAULT_TYPE):
    """
    Runs an autonomous loop, asking the agent for actions until it signals it is done or 10 minutes pass.
    """
    from telegram.constants import ChatAction
    
    # 1. Decide Strategy: Direct vs Browser
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    strategy, direct_answer = await agent.decide_strategy(user_text, chat_id)
    
    if strategy == "DIRECT":
        await context.bot.send_message(chat_id=chat_id, text=direct_answer)
        # Add to history even for direct answers
        agent.add_to_history(chat_id, user_text, [{"action": "direct_answer"}], direct_answer)
        # TTS for direct answer
        try:
            tts = gTTS(text=direct_answer[:1000], lang='en')
            audio_path = f"tts_{chat_id}.ogg"
            tts.save(audio_path)
            with open(audio_path, 'rb') as voice:
                await context.bot.send_voice(chat_id=chat_id, voice=voice)
            os.remove(audio_path)
        except Exception as e:
            logging.error(f"Error in direct TTS: {e}")
        return

    # 2. Browser Strategy
    start_time = time.time()
    max_duration = 600  # 10 minutes
    
    await context.bot.send_message(chat_id=chat_id, text=f"🌐 Browser required. Starting task: '{user_text}'.")
    
    raw_results = []
    final_response_text = ""
    is_done = False
    tts_path = None

    while time.time() - start_time < max_duration:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        screenshot = await browser.take_screenshot()
        if not screenshot:
            await browser.navigate("https://lite.duckduckgo.com/lite/")
            screenshot = await browser.take_screenshot()
            if not screenshot:
                await context.bot.send_message(chat_id=chat_id, text="Failed to start browser. Aborted.")
                return

        step_text, is_done, step_tts = await agent.analyze_and_act(user_text, screenshot, chat_id)
        raw_results.append(step_text)
        
        if not is_done:
             short_response = step_text[:200] + "..." if len(step_text) > 200 else step_text
             await context.bot.send_message(chat_id=chat_id, text=f"⏳ {short_response}")
        else:
             final_response_text = step_text
             tts_path = step_tts
             break

    if is_done:
        # 3. Refine the browser output
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        refined_answer = await agent.refine_answer(user_text, final_response_text, chat_id)
        
        # Update history with the refined answer
        agent.update_last_history_result(chat_id, refined_answer)
        
        if final_response_text.startswith("IMAGE:"):
            image_path = final_response_text.split(":", 1)[1]
            try:
                with open(image_path, 'rb') as photo:
                    await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="Here is your generated image!")
            except Exception as e:
                await context.bot.send_message(chat_id=chat_id, text=f"Error sending image: {e}")
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"✨ Final Answer:\n{refined_answer}")
            
            # If TTS was generated, use the refined text for a better experience if possible, 
            # or just send the one from analyze_and_act
            if tts_path and os.path.exists(tts_path):
                try:
                    with open(tts_path, 'rb') as voice:
                        await context.bot.send_voice(chat_id=chat_id, voice=voice)
                    os.remove(tts_path)
                except Exception as e:
                    logging.error(f"Error sending TTS: {e}")

        final_screenshot = await browser.take_screenshot()
    else:
        await context.bot.send_message(chat_id=chat_id, text="⏱️ Task timed out after 10 minutes.")

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
