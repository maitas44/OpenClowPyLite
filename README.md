# OpenClowPyLite

A lightweight, Python-based Telegram bot inspired by OpenClaw. This bot uses Playwright to browse the web, takes screenshots, and relies on Gemini Vision (`gemini-3-flash`) to understand the screen and decide on subsequent actions. It also supports image generation using `gemini-nano-banana`.

## Prerequisites

- **Python 3.13+**
- **python3-venv** (Required for the virtual environment)
  - Ubuntu/Debian: `sudo apt install python3-venv`

## Setup & Installation

1. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Get Your API Keys**
   - **Telegram Bot Token**:
     1. Open Telegram and message `@BotFather`.
     2. Send `/newbot`, choose a name and a username (ending in `bot`).
     3. Copy the HTTP API token provided (e.g., `1234567890:ABCdefGhIJKlmNoPQRstuVWXyz1234567`).
     4. Save this token inside a new file named `telegramapikey.txt` in the root of the project.
   - **Google Gemini API Key**:
     1. Go to Google AI Studio and generate an API key.
     2. Save this token inside a new file named `geminiapikey.txt` in the root of the project.

## Running the Bot

Start the bot from the command line:
```bash
python3 bot.py
```

## Usage

Find your bot in Telegram (by the username you set in BotFather) and send `/start`.

- **To browse the web**:
  ```text
  /browse https://news.ycombinator.com
  ```
  The bot will navigate to the URL, take a screenshot, and send it back to you.

- **To generate images**:
  ```text
  generate image a futuristic cyberpunk city
  ```
  The bot will use the configured image generation model to create and reply with the image.

- **To interact with the page**:
  Reply to the bot with instructions like "click the login button" or "scroll down". The Gemini Vision model will analyze the screenshot and instruct Playwright to execute the action.
