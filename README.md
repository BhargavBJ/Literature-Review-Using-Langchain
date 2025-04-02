# Research Paper Summarizer

This program automates the process of summarizing research papers by extracting key points from each section of the papers. It utilizes the Papers with Code API to search for papers based on a given query, downloads the PDFs of the papers, extracts the text from the PDFs, and generates a summary using the Google Generative AI API (gemini-2.0-flash model).

## Prerequisites

- Python 3.x
- `requests` library: `pip install requests`
- `fitz` library: `pip install pymupdf`
- `dotenv` library: `pip install python-dotenv`
- `langchain_google_genai` library: `pip install --upgrade langchain-google-genai`
- Google API Key (for the Google Generative AI API)
- PWC API Key (for the Papers with Code API)

## Setup

1. Clone the repository or download the source code.
2. Create a `.env` file in the project directory by removing the .example part in the '.env.example' file save it as `.env`

## Finally
Run the code
