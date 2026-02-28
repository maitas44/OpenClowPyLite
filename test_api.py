import asyncio
from google import genai
import os

async def test_api():
    try:
        with open("geminiapikey.txt", "r") as f:
            api_key = f.read().strip()
        print(f"Key length: {len(api_key)}")
        
        # Set environment variable EXPLICITLY before creating client
        os.environ["GOOGLE_API_KEY"] = api_key
        print("Set GOOGLE_API_KEY environment variable.")
        
        client = genai.Client(api_key=api_key)
        
        # Test synchronous list
        print("Testing Sync List...")
        try:
            models = list(client.models.list())
            print(f"Sync Success! Found {len(models)} models.")
        except Exception as e:
            print(f"Sync Failure: {e}")
            
        # Test asynchronous list
        print("Testing Async List...")
        try:
            models = await client.aio.models.list()
            print(f"Async Success! Found {len(list(models))} models.")
        except Exception as e:
            print(f"Async Failure: {e}")
            
    except Exception as e:
        print(f"General error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())
