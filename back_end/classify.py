import json
import time
from openai import OpenAI
import os
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv
load_dotenv()
# --- 1. Configure your API information ---
api_base_url = os.getenv("API_BASE_URL", "") 
api_key = os.getenv("API_KEY", "")            

# --- 2. Define helper functions and prompt templates ---

def create_llm_client(api_key, base_url):
    """Create and return an OpenAI client instance."""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        print(f"Error: Failed to create OpenAI client: {e}")
        return None

def load_json_files(input_path):
    """Load all JSON files from the input path."""
    input_dir = Path(input_path)
    json_files = {}
    
    if input_dir.is_file() and input_dir.suffix.lower() == '.json':
        # Single JSON file
        try:
            with open(input_dir, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            json_files[str(input_dir)] = papers
            print(f"Loaded single file: {input_dir}, containing {len(papers)} papers")
        except Exception as e:
            print(f"Error: Failed to load file '{input_dir}': {e}")
            
    elif input_dir.is_dir():
        # Multiple JSON files in the directory
        for json_file in input_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                json_files[str(json_file)] = papers
                print(f"Loaded file: {json_file}, containing {len(papers)} papers")
            except Exception as e:
                print(f"Error: Failed to load file '{json_file}': {e}")
    else:
        print(f"Error: Input path '{input_path}' does not exist or is not a JSON file/directory")
        
    return json_files

def prepare_data_from_dict(papers_dict):
    """Prepare data from the papers dictionary and return processed strings and paper counts."""
    processed_files = {}
    
    for file_path, papers in papers_dict.items():
        if not isinstance(papers, list):
            print(f"Warning: The content of file '{file_path}' is not a list; skipping")
            continue
            
        # Use a list to collect processed paper information
        paper_strings_list = []
        for paper in papers:
            # Only keep id, concise_description, and keywords
            paper_info = {
                "id": paper.get("id"),
                "concise_description": paper.get("concise_description"),
                "keywords": paper.get("keywords")
            }
            # Format each paper info as a JSON string and add it to the list
            paper_strings_list.append(json.dumps(paper_info, indent=2))
            
        # Use join to concatenate all strings in the list with ",\n"
        all_papers_content = ",\n".join(paper_strings_list)
        
        # Finally, wrap the concatenated content with brackets to form a complete JSON array string
        final_output_string = f"[\n{all_papers_content}\n]"
        
        processed_files[file_path] = {
            "content": final_output_string,
            "count": len(papers)
        }
    
    return processed_files

def build_classification_prompt(papers_str):
    """Build the prompt used for paper classification."""
    prompt = f"""
You are a highly intelligent AI specializing in analyzing computer science research papers. I will provide you with a list of papers, each with an ID, a concise description, and keywords.

Your task is to analyze all the papers and perform a thematic classification. Follow these instructions precisely:

1.  **Identify Themes**: Read through all the papers and identify the main research themes.
2.  **Create Categories**: Group the papers into 8 to 16 distinct categories based on these themes.
3.  **Name Categories**: Give each category a short, clear, and descriptive name (e.g., "3D Vision and NeRFs", "Object Detection in Autonomous Driving", "Vision-Language Model Analysis").
4.  **Format Output**: Provide the output as a single plain text block. For each category, list the category name followed by the IDs of the papers belonging to it. Do not include any other text, explanations, or introductory phrases like "Here is the classification:".

**Example Output Format**:
Category: 3D Vision and NeRFs
IDs: 15, 42, 133, 256

Category: Object Detection in Autonomous Driving
IDs: 3, 28, 99, 150, 312

Category: Medical Image Segmentation
IDs: 7, 81, 210

... (and so on for all categories)

Here is the list of papers to classify:
{papers_str}
"""
    return prompt

def save_classification_result(text, input_file_path, output_dir):
    """Save the classification result to the specified output directory."""
    try:
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the output file name based on the input file name
        input_path = Path(input_file_path)
        output_filename = input_path.stem + "_classification.txt"
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully saved classification result to: {output_filepath}")
        return output_filepath
    except Exception as e:
        print(f"Error: Failed to save file: {e}")
        return None

# --- 3. Main execution logic ---
def classify(
    input_path: str = r"./crawl_data/refined",
    output_dir: str = r"./crawl_data/refined_output",
    progress_cb: Optional[Callable[[int, int, float, float, str], None]] = None,
    paper_progress: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    json_files = load_json_files(input_path)
    if not json_files:
        return []

    processed_data = prepare_data_from_dict(json_files)
    client = create_llm_client(api_key, api_base_url)
    if not client:
        return []

    file_list = list(processed_data.items())
    total_files = len(file_list)
    start_ts = time.time()

    all_results = []  

    for fidx, (file_path, data_info) in enumerate(file_list, start=1):
        if paper_progress:
            paper_total = data_info["count"]
            if progress_cb:
                elapsed0 = time.time() - start_ts
                done0 = fidx - 1
                speed0 = 0 if elapsed0 == 0 else done0 / elapsed0
                eta0 = 0 if speed0 == 0 else (total_files - done0) / speed0
                progress_cb(done0, total_files, elapsed0, eta0, f"{file_path} (start)")

        classification_prompt = build_classification_prompt(data_info["content"])

        chat_completion = client.chat.completions.create(
            model="gemini-2.5-flash",  
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.3,
            max_tokens=20000,
        )
        ai_response = chat_completion.choices[0].message.content

        output_filepath = save_classification_result(ai_response, file_path, output_dir)

        all_results.append({
            "input_file": file_path,
            "output_file": output_filepath,
            "raw_classification_text": ai_response,
            "paper_count": data_info["count"]
        })
        done = fidx
        elapsed = time.time() - start_ts
        speed = 0 if elapsed == 0 else done / elapsed
        eta = 0 if speed == 0 else (total_files - done) / speed

        if progress_cb:
            progress_cb(done, total_files, elapsed, eta, file_path)

    return all_results
