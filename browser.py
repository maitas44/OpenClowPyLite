import asyncio
from playwright.async_api import async_playwright, Page, BrowserContext

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context: BrowserContext = None
        self.page: Page = None

    async def start(self):
        """Starts the Playwright browser."""
        if self.playwright:
            return

        self.playwright = await async_playwright().start()
        # Launch headless by default, but you can set headless=False for debugging
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        self.page = await self.context.new_page()

    async def stop(self):
        """Stops the browser and cleans up resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None

    async def navigate(self, url: str):
        """Navigates to the specified URL."""
        if not self.page:
            await self.start()
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def take_screenshot(self) -> bytes:
        """Takes a screenshot of the current page and returns bytes."""
        if not self.page:
            return None
        return await self.page.screenshot(type="jpeg", quality=80)

    async def get_title(self) -> str:
        if not self.page:
            return ""
        return await self.page.title()

    async def click(self, x: int, y: int):
        """Clicks at the specified coordinates."""
        if not self.page:
            return "Browser not active"
        await self.page.mouse.click(x, y)
        return f"Clicked at ({x}, {y})"

    async def type_text(self, text: str):
        """Types text into the focused element."""
        if not self.page:
            return "Browser not active"
        await self.page.keyboard.type(text)
        return f"Typed: {text}"

    async def press_key(self, key: str):
        """Presses a specific key (e.g., 'Enter', 'ArrowDown')."""
        if not self.page:
            return "Browser not active"
        await self.page.keyboard.press(key)
        return f"Pressed key: {key}"

    async def scroll(self, direction: str):
        """Scrolls the page up or down."""
        if not self.page:
            return "Browser not active"
        
        if direction == "down":
            await self.page.evaluate("window.scrollBy(0, 500)")
            return "Scrolled down"
        elif direction == "up":
            await self.page.evaluate("window.scrollBy(0, -500)")
            return "Scrolled up"
        return "Invalid scroll direction"
