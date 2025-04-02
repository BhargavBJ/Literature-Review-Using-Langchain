import os
import requests
import fitz  
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
PWC_API_KEY = os.getenv("PWC_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                             google_api_key=os.getenv("GOOGLE_API_KEY"))

DOWNLOAD_DIR = r"Path"
SUMMARY_DIR = r"Path"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

def get_papers(query, num_papers=10):
    url = f"https://paperswithcode.com/api/v1/papers/?q={query}"
    headers = {"Authorization": f"Token {PWC_API_KEY}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(" Error fetching papers:", response.json())
        return []

    data = response.json()
    print(" API Response:", data)

    papers = []
    for paper in data.get("results", [])[:num_papers]:
        title = paper.get("title", "Unknown Title")
        pdf_url = paper.get("url_pdf")  

        if pdf_url:
            papers.append((title, pdf_url))
        else:
            print(f" Skipping '{title}' (No PDF available)")

    return papers

def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url, stream=True)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f" PDF saved: {save_path}")
    else:
        print(f" Failed to download {pdf_url}")

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print(f" Error reading PDF: {pdf_path} - {e}")
        return ""

def generate_summary(pdf_text):
    prompt = f"""
    Summarize the following research paper by extracting key points from each section:
    
    Abstract:
    Summarize the abstract clearly.

    Introduction:
    Summarize the main background and problem statement.

    Methodology:
    Summarize the techniques, pre-processing, analysis, and approach used.

    Results:
    Summarize the key findings and results.

    Conclusion:
    Summarize the main takeaways and future research directions.

    Paper Text:
    {pdf_text[:10000]}
    """

    result = llm.invoke(prompt)
    return result.content

def process_papers(query):
    papers = get_papers(query)

    if not papers:
        print(" No papers with PDFs found.")
        return

    for idx, (title, pdf_url) in enumerate(papers):
        pdf_path = os.path.join(DOWNLOAD_DIR, f"paper_{idx+1}.pdf")
        text_path = os.path.join(SUMMARY_DIR, f"summary_{idx+1}.txt")

        download_pdf(pdf_url, pdf_path)

        pdf_text = extract_text_from_pdf(pdf_path)

        if not pdf_text.strip():
            print(f" No text extracted from {pdf_path}")
            continue

        summary = generate_summary(pdf_text)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n\n")
            f.write(summary)

        print(f" Summary saved to: {text_path}")

process_papers("Reinforcement learning")
