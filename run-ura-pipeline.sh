#!/bin/bash
set -e

# ASCII art banner
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 AMIL VOICE-TO-URA PIPELINE                 ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Check if ELEVENLABS_API_KEY is set
if [ -z "$ELEVENLABS_API_KEY" ]; then
  echo "❌ Error: ELEVENLABS_API_KEY environment variable is not set."
  echo "Please set it with: export ELEVENLABS_API_KEY=your_key_here"
  exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "❌ Error: Docker is not installed. Please install Docker first."
  exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
  echo "❌ Error: Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

# Stop any existing containers
echo "🔄 Stopping any existing containers..."
docker compose down

# Start the services
echo "🚀 Starting URA Pipeline services..."
docker compose up -d

# Wait for the service to be ready
echo "⏳ Waiting for URA Pipeline service to be ready..."
for i in {1..15}; do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ URA Pipeline is ready!"
    echo "🌐 Access the web interface at: http://localhost:8000"
    echo
    echo "📋 Commands:"
    echo "  - View logs: docker compose logs -f"
    echo "  - Stop service: docker compose down"
    exit 0
  fi
  echo "Waiting for service... ($i/15)"
  sleep 2
done

echo "⚠️  Service did not start in the expected time. Check logs with: docker compose logs"
echo "You can still try accessing http://localhost:8000 manually." 