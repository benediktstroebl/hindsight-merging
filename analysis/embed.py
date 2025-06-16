import random
import os
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import glob
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
import re 

load_dotenv()

def extract_content(text):
    pattern = r'```(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def truncate_text(text, max_tokens=8000):
    text = extract_content(text)
    if text is None:
        return None 
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=2, timeout=10.0)

def embed_text(text: str, dimensions: int = 1024) -> List[float]:
    """
    Generate embeddings for a given text using OpenAI's API.
    
    Args:
        text (str): Input text to embed
        dimensions (int): Number of dimensions for the embedding
        
    Returns:
        List[float]: Embedding vector
    """
    if text is None:
        return None 
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=dimensions
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_file(file_path: str, output_dir: str, max_tokens: int = None, overwrite: bool = False) -> Dict:
    """
    Process a single JSON file and generate embeddings for both text fields.
    Only process files with 'converted' prefix and add missing embeddings.
    
    Args:
        file_path (str): Path to the JSON file
        output_dir (str): Directory where output files are saved
        
    Returns:
        Dict: Dictionary containing the original data and its embeddings, or None if skipped
    """
    # Only process files with 'converted' prefix
    # if not os.path.basename(file_path).startswith('converted'):
    #     return None
    
    question_num = int(file_path.split('question_')[1].split('_')[0])
    # if question_num > 999:
    #     pass 
    
    try:
        # Check if output file exists and load it
        if max_tokens:
            output_path = os.path.join(
                output_dir,
                os.path.basename(file_path).replace('.json', f'_with_embeddings.json')
            )
        else:
            output_path = os.path.join(
                output_dir,
                os.path.basename(file_path).replace('.json', '_with_embeddings.json')
            )
        
        data = None
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
        # Check for required fields
        if 'output' not in data:
            print(f"Warning: 'generated_text' not found in {file_path}")
            return None
            
        # Generate missing embeddings
        if 'generated_embedding' not in data:
            generated_embedding = embed_text(truncate_text(data['output']))
            if generated_embedding:
                data['generated_embedding'] = generated_embedding
            
        if ('converted_text' in data) and ('converted_embedding' not in data) or overwrite:
            converted_embedding = embed_text(truncate_text(data['converted_text'], max_tokens))
            if converted_embedding:
                data['converted_embedding'] = converted_embedding
        
        return data if ('generated_embedding' in data or 'converted_embedding' in data) else None
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_directory(input_dir: str, output_dir: str, max_workers: int = 4, max_tokens: int = None, sample_size: int = None, overwrite: bool = False):
    """
    Process all JSON files in a directory in parallel.
    
    Args:
        input_dir (str): Directory containing input JSON files
        output_dir (str): Directory to save processed files
        max_workers (int): Maximum number of parallel workers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    question_nums = [int(os.path.basename(file_path).split('question_')[1].split('_')[0]) for file_path in glob.glob(os.path.join(input_dir, "*.json"))]
    sample_question_nums = random.sample(list(set(question_nums)), sample_size) if sample_size else question_nums
    if sample_size:
        json_files = [f for f in glob.glob(os.path.join(input_dir, "*.json")) if int(os.path.basename(f).split('question_')[1].split('_')[0]) in sample_question_nums]
    else:
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(json_files)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Processing {len(json_files)} files...")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress bar
        results = list(tqdm(
            executor.map(lambda x: process_file(x, output_dir, max_tokens, overwrite), json_files),
            total=len(json_files),
            desc="Generating embeddings"
        ))
    
    # Save processed files
    processed_count = 0
    for file_path, result in zip(json_files, results):
        if result:
            if max_tokens:
                output_path = os.path.join(
                    output_dir,
                    os.path.basename(file_path).replace('.json', f'_with_embeddings.json')
                )
            else:
                output_path = os.path.join(
                    output_dir,
                    os.path.basename(file_path).replace('.json', '_with_embeddings.json')
                )
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            processed_count += 1
    
    print(f"Processed {processed_count} files with new or updated embeddings.")

if __name__ == "__main__":
    # Example usage
    # run_name = "taco_instruct_llama_8b_single_slerp_0.5/"
    input_directory = f"/scratch/gpfs/bs6865/messi-thinking/SkyThought/samples/"
    run_names = glob.glob(input_directory + '*_nosysprompt')
    print(run_names)
    # input_directory = f"/scratch/gpfs/vv7118/projects/messi-thinking/{run_name}"
    for run_name in tqdm(run_names):
        fname = run_name.split('/')[-1]
        output_directory = f"embeddings/{fname}"
        os.makedirs(output_directory, exist_ok=True)
        
        process_directory(
            run_name,
            output_directory,
            max_workers=10  # Adjust based on your needs
        )
