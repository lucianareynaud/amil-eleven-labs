#!/usr/bin/env python3
import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_elevenlabs_synthesis():
    """Test ElevenLabs API integration with a simple phrase"""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "ErXwobaYiN019PkySvjV")  # Antoni (male) for pt-BR
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v1")
    stability = float(os.getenv("ELEVENLABS_STABILITY", "0.50"))
    similarity_boost = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
    
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY environment variable not set.")
        print("Please set it in .env file or export it in your shell.")
        sys.exit(1)
    
    # Test text for synthesis
    test_text = "<speak>Bem-vindo ao teste de URA.<break time=\"400ms\"/> Esta é uma mensagem de teste para verificar a integração com ElevenLabs.</speak>"
    
    # ElevenLabs API parameters
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
    headers = {
        'xi-api-key': api_key,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg',
    }
    payload = {
        'text': test_text,
        'model_id': model_id,
        'voice_settings': {'stability': stability, 'similarity_boost': similarity_boost}
    }
    
    print("Testing ElevenLabs API integration...")
    print(f"Using voice ID: {voice_id}")
    print(f"Using model: {model_id}")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Save the audio file
        output_file = "test_output.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"SUCCESS: Audio generated and saved to {output_file}")
        print(f"File size: {len(response.content)} bytes")
        return True
    
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        if e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Connection error. Please check your internet connection.")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. The ElevenLabs API might be experiencing delays.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred: {e}")
    
    return False

if __name__ == "__main__":
    success = test_elevenlabs_synthesis()
    sys.exit(0 if success else 1) 