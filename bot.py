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

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_authorized(chat_id):
        await context.bot.send_message(chat_id=chat_id, text="Unauthorized access.")
        return

    await context.bot.send_message(chat_id=chat_id, text="🧹 Resetting browser session... This will clear all cookies and history.")
    await browser.stop()
    await browser.start() # Ensure it's ready for the next action
    await context.bot.send_message(chat_id=chat_id, text="✨ Browser session has been reset! You now have a blank slate.")
    
async def _solve_autonomous(chat_id: int, user_text: str, context: ContextTypes.DEFAULT_TYPE, user_image_path: str = None):
    """
    Runs an autonomous loop, asking the agent for actions until it signals it is done or 10 minutes pass.
    """
    from telegram.constants import ChatAction
    
    # 1. Decide Strategy: Direct vs Browser vs Image
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    strategy, direct_answer, image_prompt = await agent.decide_strategy(user_text, chat_id)
    
    final_output_text = ""
    final_image_path = None
    
    if strategy == "DIRECT":
        final_output_text = direct_answer
        agent.add_to_history(chat_id, user_text, [{"action": "direct_answer"}], direct_answer)

    elif strategy == "IMAGE":
        await context.bot.send_message(chat_id=chat_id, text="🎨 Generating image...")
        result = await agent.generate_image(image_prompt)
        if result.startswith("IMAGE:"):
            final_image_path = result.split(":", 1)[1]
            final_output_text = f"Image generated: {final_image_path}"
        else:
            final_output_text = result
        agent.add_to_history(chat_id, user_text, [{"action": "generate_image"}], final_output_text)

    elif strategy == "BROWSER":
        # 2. Browser Strategy
        max_duration = 600  # 10 minutes
        max_retries = 2
        retry_count = 0
        success = False
        
        while retry_count <= max_retries and not success:
            start_time = time.time()
            if retry_count == 0:
                await context.bot.send_message(chat_id=chat_id, text=f"🌐 Browser required for: '{user_text}'.")
            else:
                await context.bot.send_message(chat_id=chat_id, text=f"🔄 Verification failed. Retrying (Attempt {retry_count + 1}/{max_retries + 1})...")

            # Reset the step journal so this task starts with a clean history
            agent.reset_task_steps()
            
            is_done = False
            step_tts_path = None
            browser_raw_answer = ""
            
            # Capture start state
            start_url = await browser.get_url()
            start_title = await browser.get_title()

            while time.time() - start_time < max_duration:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                
                # Draw SoM labels before taking the screenshot for Gemini
                await browser.draw_som()
                screenshot = await browser.take_screenshot()
                # Remove labels immediately so they don't interfere with the page state
                await browser.remove_som()

                if not screenshot:
                    await browser.navigate("https://lite.duckduckgo.com/lite/")
                    await browser.draw_som()
                    screenshot = await browser.take_screenshot()
                    await browser.remove_som()
                    if not screenshot:
                        await context.bot.send_message(chat_id=chat_id, text="Failed to start browser.")
                        return

                step_text, is_done, step_tts = await agent.analyze_and_act(user_text, screenshot, chat_id, user_image_path)
                if not is_done:
                     short_response = step_text[:200] + "..." if len(step_text) > 200 else step_text
                     await context.bot.send_message(chat_id=chat_id, text=f"⏳ {short_response}")
                     # Send current browser screenshot so user can see what's happening
                     step_screenshot = await browser.take_screenshot()
                     if step_screenshot:
                         try:
                             await context.bot.send_photo(chat_id=chat_id, photo=step_screenshot,
                                                           caption=f"🌐 Browser state (step {len(agent._task_steps)})")
                         except Exception as photo_err:
                             print(f"Failed to send step screenshot: {photo_err}")

                else:
                     browser_raw_answer = step_text
                     step_tts_path = step_tts # Store for later
                     break
            
            if is_done:
                # Capture end state
                end_url = await browser.get_url()
                end_title = await browser.get_title()

                # 3. Refine the browser output
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                final_output_text = await agent.refine_answer(user_text, browser_raw_answer, chat_id)
                agent.update_last_history_result(chat_id, final_output_text)
                
                # Special check for image result from browser loop (if applicable)
                if browser_raw_answer.startswith("IMAGE:"):
                    final_image_path = browser_raw_answer.split(":", 1)[1]
            else:
                await context.bot.send_message(chat_id=chat_id, text="⏱️ Task timed out.")
                return

            # 4. Final Verification Step
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            
            # Use states for verification context
            verification_context = f"Start Page: {start_title} ({start_url})\nEnd Page: {end_title} ({end_url})"
            
            success, feedback = await agent.verify_result(f"{user_text}\n\nCONTEXT:\n{verification_context}", final_output_text, final_image_path, user_image_path)
            
            # Store verification results in session history
            agent.update_verification_to_history(chat_id, success, feedback)
            
            if not success:
                retry_count += 1
                if retry_count <= max_retries:
                    await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Verification Failed: {feedback}\nI will adjust my plan and try again.")
                    # Trigger replanning with feedback
                    if agent.current_plan:
                         agent.current_plan = await agent.planner.update_plan(agent.current_plan, len(agent._task_steps), f"Verification failed: {feedback}")
                else:
                    await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Verification Failed after {max_retries} retries: {feedback}\nDelivering result anyway.")
                    break # Break to deliver final result
            else:
                break # Success, proceed to delivery

    # 5. Final Delivery
    if final_image_path:
        try:
            with open(final_image_path, 'rb') as photo:
                await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="Here is your requested image!")
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"Error sending image: {e}")
    else:
        await context.bot.send_message(chat_id=chat_id, text=f"✨ {final_output_text}")
        
    # Final TTS (for non-image answers)
    if not final_image_path:
        try:
            from gtts import gTTS
            tts = gTTS(text=final_output_text[:1000], lang='en')
            audio_path = f"tts_{chat_id}.voice"
            tts.save(audio_path)
            with open(audio_path, 'rb') as voice:
                await context.bot.send_voice(chat_id=chat_id, voice=voice)
            os.remove(audio_path)
        except Exception as e:
            logging.error(f"Error in TTS: {e}")

    # Optional final screenshot
    if strategy == "BROWSER" and not final_image_path:
        final_screenshot = await browser.take_screenshot()
        if final_screenshot:
            await context.bot.send_photo(chat_id=chat_id, photo=final_screenshot)

    # Clean up user image if it was used
    if user_image_path and os.path.exists(user_image_path):
        try:
            os.remove(user_image_path)
            if 'user_image_path' in context.user_data:
                del context.user_data['user_image_path']
        except:
            pass

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

    # Natural language session reset detection
    reset_keywords = ["reset session", "blank session", "clear session", "reset browser", "new session"]
    if any(kw in user_text.lower() for kw in reset_keywords):
        await context.bot.send_message(chat_id=chat_id, text="🧹 Natural language reset detected. Clearing browser session...")
        await browser.stop()
        await browser.start()
        await context.bot.send_message(chat_id=chat_id, text="✨ Session reset complete. What's next?")
        return

    # Check if we have a pending image from a previous upload
    user_image_path = context.user_data.get('user_image_path')

    # Hand off to the autonomous loop
    asyncio.create_task(_solve_autonomous(chat_id, user_text, context, user_image_path))

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_authorized(chat_id):
        await context.bot.send_message(chat_id=chat_id, text="Unauthorized access.")
        return

    global last_activity_time, needs_improvement, last_chat_id
    last_activity_time = time.time()
    needs_improvement = True
    last_chat_id = chat_id
    
    photo_file = await update.message.photo[-1].get_file()
    user_image_path = f"user_photo_{chat_id}.jpg"
    await photo_file.download_to_drive(user_image_path)
    
    context.user_data['user_image_path'] = user_image_path
    
    caption = update.message.caption
    if caption:
        await context.bot.send_message(chat_id=chat_id, text="📸 Image received with caption. Processing...")
        asyncio.create_task(_solve_autonomous(chat_id, caption, context, user_image_path))
    else:
        await context.bot.send_message(chat_id=chat_id, text="📸 I've received your image! What would you like me to do with it?")

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
    
    # Check if we have a pending image
    user_image_path = context.user_data.get('user_image_path')
    
    # Hand off to autonomous logic
    asyncio.create_task(_solve_autonomous(chat_id, transcript, context, user_image_path))

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
    reset_handler = CommandHandler('reset', reset_command)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    audio_handler = MessageHandler(filters.VOICE | filters.AUDIO, handle_audio)
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)

    application.add_handler(start_handler)
    application.add_handler(browse_handler)
    application.add_handler(reset_handler)
    application.add_handler(message_handler)
    application.add_handler(audio_handler)
    application.add_handler(photo_handler)

    # Start the background job to monitor inactivity every 5 seconds
    application.job_queue.run_repeating(check_inactivity, interval=5)

    print("Bot is polling...")
    application.run_polling()
