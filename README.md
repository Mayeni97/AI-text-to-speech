# AI Text to Speech

A Python application that converts text to speech using AI models. It summarizes input text and generates natural-sounding speech output.

## Project Structure

```
.  
├── models/         # Directory for AI model files
├── output/        # Directory for generated audio files
├── src/           # Source code
│   └── app.py     # Main application file
└── utils/         # Utility functions
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with the following variables:
```
MODELS_PATH=models
OUTPUT_PATH=output
```

## Usage

Run the application:
```bash
python src/app.py
```

The application provides:
- Text input or file upload (supports txt, pdf, doc, docx)
- AI-powered text summarization
- High-quality speech synthesis
- Audio playback and download options