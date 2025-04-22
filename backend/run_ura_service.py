#!/usr/bin/env python3
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check for ElevenLabs API key
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("WARNING: ELEVENLABS_API_KEY environment variable not set.")
        print("Voice synthesis will not work without this key.")
        print("Please set it in your .env file or export it in your shell.")
    
    # Get host and port from environment or use defaults
    host = os.getenv("URA_HOST", "0.0.0.0")
    port = int(os.getenv("URA_PORT", "8000"))
    
    print(f"Starting URA Pipeline service on {host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the FastAPI application
    uvicorn.run(
        "ura_pipeline:app",
        host=host,
        port=port,
        reload=True
    ) 