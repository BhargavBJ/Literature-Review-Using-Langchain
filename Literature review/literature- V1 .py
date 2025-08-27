import os
import time
import requests
import fitz  
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ===============================
# Load ENV
# ===============================
load_dotenv()
token = os.getenv("CORE_API")  # renamed as requested
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))

DOWNLOAD_DIR = r"Path"
SUMMARY_DIR = r"Path"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


# ===============================
# CORE API fetch
# ===============================
def get_papers_core(query, num_papers=10, retries=3):
    url = f"https://api.core.ac.uk/v3/search/works/?q={query}&limit={num_papers}"
    headers = {"Authorization": f"Token {token}"}

    for attempt in range(retries):
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(" Error fetching papers:", response.text)
            time.sleep(2)
            continue

        data = response.json()
        results = data.get("results", [])

        if not results:
            print(" CORE API returned no results.")
            return []

        papers = []
        for paper in results:
            title = paper.get("title", "Unknown Title")
            pdf_url = paper.get("downloadUrl") or paper.get("url") or None

            if pdf_url:
                papers.append((title, pdf_url))
            else:
                print(f" Skipping '{title}' (No PDF available)")

        if papers:
            return papers

    return []


# ===============================
# arXiv fallback
# ===============================
def get_papers_arxiv(query, num_papers=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={num_papers}"
    response = requests.get(url)

    if response.status_code != 200:
        print(" Error fetching from arXiv:", response.text)
        return []

    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.content)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
                break
        if pdf_url:
            papers.append((title, pdf_url))

    return papers


# ===============================
# Download PDF
# ===============================
def download_pdf(pdf_url, save_path):
    try:
        response = requests.get(pdf_url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" PDF saved: {save_path}")
            return True
    except Exception as e:
        print(f" Failed to download {pdf_url} -> {e}")
    return False


# ===============================
# Extract text
# ===============================
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print(f" Error reading PDF: {pdf_path} - {e}")
        return ""


# ===============================
# Summarization
# ===============================
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

    References:
    Just give the references as is

    Paper Text:
    {pdf_text[:10000]}
    """

    result = llm.invoke(prompt)
    return result.content


# ===============================
# Main Processing
# ===============================
def process_papers(query):
    papers = get_papers_core(query)
    if not papers:
        print("CORE failed, switching to arXiv fallback...")
        papers = get_papers_arxiv(query)

    if not papers:
        print("No papers found on CORE or arXiv.")
        return

    for idx, (title, pdf_url) in enumerate(papers):
        pdf_path = os.path.join(DOWNLOAD_DIR, f"paper_{idx+1}.pdf")
        text_path = os.path.join(SUMMARY_DIR, f"summary_{idx+1}.txt")

        if not download_pdf(pdf_url, pdf_path):
            continue

        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            print(f" No text extracted from {pdf_path}")
            continue

        summary = generate_summary(pdf_text)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n\n")
            f.write(summary)

        print(f"Summary saved to: {text_path}")


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    process_papers("Whatever topic you want to research and study on")

