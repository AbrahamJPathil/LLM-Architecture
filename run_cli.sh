#!/bin/bash
# Helper script to run PromptEvolve CLI with proper environment

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment from .env file...${NC}"
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo -e "${YELLOW}Please set OPENAI_API_KEY environment variable${NC}"
    echo ""
    echo "You can either:"
    echo "  1. Create a .env file with: OPENAI_API_KEY=your-key-here"
    echo "  2. Export it: export OPENAI_API_KEY=your-key-here"
    echo ""
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    echo "Please set your OpenAI API key before running."
    exit 1
fi

# Run the CLI
echo -e "${GREEN}Running PromptEvolve CLI...${NC}\n"
uv run python promptevolve.py "$@"
