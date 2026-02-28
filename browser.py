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
            viewport={"width": 1920, "height": 1080},
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

    async def get_url(self) -> str:
        """Returns the current page URL."""
        if not self.page:
            return ""
        return self.page.url

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

    async def fill_field(self, x: int, y: int, text: str):
        """Clicks a field at (x,y), selects all existing content with Ctrl+A, then
        types the new text — replacing whatever was already in the field."""
        if not self.page:
            return "Browser not active"
        # Triple-click selects the entire field value across all input types
        await self.page.mouse.click(x, y, click_count=3)
        await self.page.keyboard.press("Control+a")
        await self.page.keyboard.type(text)
        return f"Filled field at ({x}, {y}) with: {text}"

    async def press_key(self, key: str):
        """Presses a specific key (e.g., 'Enter', 'ArrowDown')."""
        if not self.page:
            return "Browser not active"
        await self.page.keyboard.press(key)
        return f"Pressed key: {key}"

    async def get_text_content(self) -> str:
        """Extracts and returns all visible text from the current page body."""
        if not self.page:
            return "Browser not active"
        try:
            return await self.page.inner_text("body")
        except Exception as e:
            return f"Error extracting text: {e}"

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

    async def fill_by_placeholder(self, placeholder: str, text: str) -> str:
        """Finds an input by its placeholder text and fills it. Clears first.
        PREFERRED over coordinate-based clicks for form fields."""
        if not self.page:
            return "Browser not active"
        try:
            locator = self.page.get_by_placeholder(placeholder, exact=False)
            await locator.first.clear()
            await locator.first.fill(text)
            return f"Filled placeholder='{placeholder}' with: {text}"
        except Exception as e:
            return f"Error filling by placeholder '{placeholder}': {e}"

    async def fill_by_label(self, label: str, text: str) -> str:
        """Finds an input by its associated label text and fills it. Clears first.
        PREFERRED over coordinate-based clicks for form fields."""
        if not self.page:
            return "Browser not active"
        try:
            locator = self.page.get_by_label(label, exact=False)
            await locator.first.clear()
            await locator.first.fill(text)
            return f"Filled label='{label}' with: {text}"
        except Exception as e:
            return f"Error filling by label '{label}': {e}"

    async def click_by_text(self, text: str) -> str:
        """Clicks a button or link by its visible text content.
        PREFERRED over coordinate-based clicks for buttons."""
        if not self.page:
            return "Browser not active"
        try:
            # Try button first, then any role
            locator = self.page.get_by_role("button", name=text, exact=False)
            count = await locator.count()
            if count == 0:
                locator = self.page.get_by_text(text, exact=False)
            await locator.first.click()
            return f"Clicked button/link with text='{text}'"
        except Exception as e:
            return f"Error clicking by text '{text}': {e}"

    async def get_form_fields(self) -> str:
        """Returns a JSON list of all visible form inputs with their placeholder, label, type, and id.
        Use this to discover field names before filling a form."""
        if not self.page:
            return "Browser not active"
        try:
            fields = await self.page.evaluate("""() => {
                const inputs = Array.from(document.querySelectorAll('input, select, textarea'));
                return inputs.map(el => ({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || '',
                    id: el.id || '',
                    name: el.name || '',
                    placeholder: el.placeholder || '',
                    value: el.value || '',
                    label: (() => {
                        if (el.id) {
                            const lbl = document.querySelector('label[for="' + el.id + '"]');
                            return lbl ? lbl.innerText.trim() : '';
                        }
                        return '';
                    })()
                }));
            }""")
            import json
            return json.dumps(fields, ensure_ascii=False)
        except Exception as e:
            return f"Error getting form fields: {e}"

    async def draw_som(self):
        """Injects numbered labels (Set of Marks) over all interactive elements."""
        if not self.page:
            return
        await self.page.evaluate("""() => {
            const interactives = Array.from(document.querySelectorAll('a, button, input, select, textarea, [role="button"], [onclick]'));
            interactives.forEach((el, i) => {
                const rect = el.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top <= window.innerHeight) {
                    const label = document.createElement('div');
                    label.className = 'openclaw-som-label';
                    label.innerText = i;
                    label.style.position = 'fixed';
                    label.style.top = rect.top + 'px';
                    label.style.left = rect.left + 'px';
                    label.style.backgroundColor = 'red';
                    label.style.color = 'white';
                    label.style.fontSize = '12px';
                    label.style.fontWeight = 'bold';
                    label.style.padding = '2px 4px';
                    label.style.borderRadius = '3px';
                    label.style.zIndex = '1000000';
                    label.style.pointerEvents = 'none';
                    document.body.appendChild(label);
                    el.setAttribute('data-openclaw-id', i);
                }
            });
        }""")

    async def remove_som(self):
        """Removes all SoM labels injected by draw_som."""
        if not self.page:
            return
        await self.page.evaluate("""() => {
            const labels = document.querySelectorAll('.openclaw-som-label');
            labels.forEach(l => l.remove());
        }""")

    async def click_by_id(self, som_id: int):
        """Clicks an element by its SoM ID."""
        if not self.page:
            return "Browser not active"
        try:
            await self.page.click(f'[data-openclaw-id="{som_id}"]')
            return f"Clicked element #{som_id}"
        except Exception as e:
            return f"Error clicking element #{som_id}: {e}"

    async def fill_by_id(self, som_id: int, text: str):
        """Fills an input by its SoM ID."""
        if not self.page:
            return "Browser not active"
        try:
            locator = self.page.locator(f'[data-openclaw-id="{som_id}"]')
            await locator.clear()
            await locator.fill(text)
            return f"Filled element #{som_id} with: {text}"
        except Exception as e:
            return f"Error filling element #{som_id}: {e}"

    async def get_accessibility_snapshot(self) -> str:
        """Returns a simplified DOM tree of interactive elements."""
        if not self.page:
            return ""
        try:
            return await self.page.evaluate("""() => {
                const interactives = Array.from(document.querySelectorAll('a, button, input, select, textarea, [role="button"], [onclick], [data-openclaw-id]'));
                return interactives.map(el => {
                    const tag = el.tagName ? el.tagName.toLowerCase() : '';
                    const role = el.getAttribute('role') || tag;
                    const text = (el.innerText || el.value || el.placeholder || '').slice(0, 100).trim().replace(/\\n/g, ' ');
                    const somId = el.getAttribute('data-openclaw-id') || 'none';
                    return `[ID: ${somId}] ${role}: "${text}"`;
                }).filter(s => s.indexOf('""') === -1).join('\\n');
            }""")
        except Exception as e:
            return f"Error taking accessibility snapshot: {e}"

