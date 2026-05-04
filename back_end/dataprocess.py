import os
import json
import time
from pathlib import Path
from typing import Callable, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_base_url = os.getenv("API_BASE_URL", "")
api_key = os.getenv("API_KEY", "")

# =========================
# 1. Arxiv JSON processing
# =========================

def _process_single_arxiv_json(file_path: Path):
    """
    Read and process a single arxiv_output.json file and organize paper information
    into a list in the specified format.

    Parameters:
    file_path (Path): Path object pointing to a JSON file.

    Returns:
    list: A list of strings containing the formatted paper information.
          If the file does not exist or the format is incorrect, an empty list is returned.
    """
    formatted_papers = []

    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: file '{file_path}' is not a valid JSON file.")
        return []

    # Check whether the 'papers' key exists and is a list
    if 'papers' not in data or not isinstance(data['papers'], list):
        print(
            f"Error: the 'papers' key is missing or incorrectly formatted "
            f"in the JSON file '{file_path}'."
        )
        return []

    # Iterate over each paper and format it
    for index, paper in enumerate(data['papers']):
        title = paper.get('title', 'N/A')
        abstract = paper.get('abstract', 'N/A')

        # Concatenate the string in the format "no.(index) title:(...) abstract:(...)"
        formatted_string = f"no.{index + 1} title:{title} abstract:{abstract}"
        formatted_papers.append(formatted_string)

    return formatted_papers


def process_arxiv_json(path_str: str):
    """
    If given a file path: return a list of formatted papers.
    If given a directory path: process all *.json files under this directory and
    return a dict: {json_file_path: [formatted_papers]}.

    Parameters:
    path_str (str): Path to a JSON file or a directory containing JSON files.

    Returns:
    list or dict:
        - list[str]: if path_str is a single JSON file.
        - dict[str, list[str]]: if path_str is a directory containing JSON files.
    """
    path = Path(path_str)

    # Case 1: single JSON file
    if path.is_file():
        return _process_single_arxiv_json(path)

    # Case 2: directory – process all JSON files in this directory
    if path.is_dir():
        result = {}
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            print(f"Warning: no JSON files found under directory '{path}'.")
            return {}

        for json_file in json_files:
            print(f"Processing JSON file: {json_file}")
            papers = _process_single_arxiv_json(json_file)
            if papers:
                # one JSON file corresponds to one output list
                result[str(json_file)] = papers
        return result

    print(f"Error: '{path_str}' is neither a file nor a directory.")
    return []


# ==========================================
# 2. LLM config, helpers and prompt
# ==========================================
def create_llm_client(api_key: str, base_url: str):
    """Create and return an OpenAI client instance."""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        print(f"Error: failed to create OpenAI client: {e}")
        return None


def build_batch_prompt(papers_batch):
    """Build the prompt to send to the LLM for a batch of papers."""
    
    # Format the papers in the batch and index them
    papers_str = ""
    for i, paper_info in enumerate(papers_batch):
        papers_str += f"Paper {i+1}:\n{paper_info}\n\n"

    prompt = f"""
You are an expert AI assistant specializing in summarizing academic papers in computer vision.
I will provide you with a batch of papers. For each paper in the batch, perform the following two tasks:

1.  **Concise Description**: Write a single, highly concise sentence that summarizes the paper's core contribution. It should clearly state the problem, the proposed method, or the key outcome.
2.  **Keywords**: Generate 4-5 keywords that best represent the paper's main topics and techniques.

Please process all papers provided below and return the result as a single JSON array. Each object in the array should correspond to a paper in the order it was given and must contain 'id', 'concise_description', and 'keywords' fields.

**Input Papers**:
{papers_str}

**Output Format Example**:
[
  {{
    "id": 1,
    "concise_description": "A one-sentence summary of the first paper.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
  }},
  {{
    "id": 2,
    "concise_description": "A one-sentence summary of the second paper.",
    "keywords": ["keywordA", "keywordB", "keywordC", "keywordD"]
  }}
]

**IMPORTANT**: Ensure your output is a valid JSON array string and nothing else. Do not include any introductory text like "Here is the JSON output:".
"""
    return prompt


def process_batch(client, batch, model: str = "gemini-2.5-flash", max_retries: int = 3):
    """Send a single batch to the LLM and process the returned results, with a retry mechanism."""
    prompt = build_batch_prompt(batch)
    
    for attempt in range(max_retries):
        try:
            print(f"Sending batch (containing {len(batch)} papers)... Attempt: {attempt + 1}/{max_retries}")
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI assistant that strictly follows user instructions and outputs only valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                # response_format={"type": "json_object"},
            )
            
            ai_response_text = chat_completion.choices[0].message.content
            
            # Try to parse JSON
            if ai_response_text.strip().startswith("```json"):
                ai_response_text = ai_response_text.strip()[7:-3].strip()

            parsed_json = json.loads(ai_response_text)
            
            # Verify that the returned JSON is a list and that the number of entries matches
            if isinstance(parsed_json, list) and len(parsed_json) == len(batch):
                print(f"Batch processed successfully; received {len(parsed_json)} records.")
                return parsed_json
            else:
                print(
                    "Warning: JSON format is valid, but content is not as expected. "
                    f"Returned {len(parsed_json)} records; expected {len(batch)}."
                )
                
        except json.JSONDecodeError:
            print("Error: failed to parse JSON. The LLM response is not valid JSON.")
            print("LLM response:", ai_response_text)
        except Exception as e:
            print(f"Error: API request failed: {e}")
        
        print("Processing failed; retrying in 5 seconds...")
        time.sleep(5)
        
    print(f"Batch processing failed; maximum number of retries reached ({max_retries}).")
    return None  # Return None to indicate that this batch failed


def save_to_json(data, filename: str):
    """Save data to a file in JSON format."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to file: {filename}")
    except Exception as e:
        print(f"Error: failed to save file '{filename}': {e}")


# =========================
# 3. Per-file refinement
# =========================

def refine_papers_for_one_file(
    paper_list,
    output_filename: str,
    client,
    batch_size: int = 20,
    progress_cb: Optional[Callable[[int, int, float, float, str], None]] = None,
    source_json_path: Optional[Path] = None,  
):
    if not paper_list:
        print(f"No papers to process for output file '{output_filename}'. Skip.")
        return []
    source_papers = None
    if source_json_path is not None and source_json_path.is_file():
        try:
            with source_json_path.open("r", encoding="utf-8") as f:
                src_data = json.load(f)
            if isinstance(src_data, dict) and isinstance(src_data.get("papers"), list):
                source_papers = src_data["papers"]
            else:
                print(f"Warning: 'papers' not found or invalid in {source_json_path}")
        except Exception as e:
            print(f"Warning: failed to load source json {source_json_path}: {e}")

    all_processed_papers = []
    total = len(paper_list)
    start_ts = time.time()

    for i in range(0, total, batch_size):
        batch = paper_list[i: i + batch_size]
        processed_batch = process_batch(client, batch)
        if processed_batch:
            for j, (original_paper, refined_info) in enumerate(
                zip(batch, processed_batch), start=i
            ):
                refined_info["original_text"] = original_paper
                if source_papers is not None and 0 <= j < len(source_papers):
                    src = source_papers[j]
                    refined_info["html_url"] = src.get("html_url")
                    refined_info["arxiv_id"] = src.get("arxiv_id")
                    refined_info["section"] = src.get("section")

                all_processed_papers.append(refined_info)
        done = min(i + len(batch), total)
        elapsed = time.time() - start_ts
        speed = 0 if elapsed == 0 else done / elapsed
        eta = 0 if speed == 0 else (total - done) / speed

        if progress_cb:
            progress_cb(done, total, elapsed, eta, str(output_filename))

    save_to_json(all_processed_papers, output_filename)
    return all_processed_papers


# =========================
# 4. Main execution logic
# =========================
def data_process(progress_cb: Optional[Callable[[int, int, float, float, str], None]] = None):
    input_path = "./crawl_data"  
    result = process_arxiv_json(input_path)

    client = create_llm_client(api_key, api_base_url)
    if not client:
        return []

    all_refined = []
    if isinstance(result, list):
        output_dir = "./crawl_data/refined"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, "refined_papers.json")

        source_path = Path(input_path) if Path(input_path).is_file() else None

        all_refined = refine_papers_for_one_file(
            result,
            output_filename,
            client,
            progress_cb=progress_cb,
            source_json_path=source_path,
        )
    elif isinstance(result, dict):
        output_dir = "./crawl_data/refined"
        os.makedirs(output_dir, exist_ok=True)

        for file_path, paper_list in result.items():
            p = Path(file_path)
            output_filename = os.path.join(output_dir, p.stem + "_refined.json")
            refined = refine_papers_for_one_file(
                paper_list,
                output_filename,
                client,
                progress_cb=progress_cb,
                source_json_path=p,  
            )
            all_refined.extend(refined)

    return all_refined
