import os
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import glob
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

load_dotenv()

def truncate_text(text, max_tokens=8000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str, dimensions: int = 150) -> List[float]:
    """
    Generate embeddings for a given text using OpenAI's API.
    
    Args:
        text (str): Input text to embed
        dimensions (int): Number of dimensions for the embedding
        
    Returns:
        List[float]: Embedding vector
    """
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

def process_file(file_path: str, output_dir: str) -> Dict:
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
    if not os.path.basename(file_path).startswith('converted'):
        return None
    
    question_num = int(file_path.split('question_')[1].split('_')[0])
    if question_num > 499:
        return None 
    
    try:
        # Check if output file exists and load it
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
        if 'generated_text' not in data:
            print(f"Warning: 'generated_text' not found in {file_path}")
            return None
            
        # Generate missing embeddings
        if 'generated_embedding' not in data:
            generated_embedding = embed_text(truncate_text(data['generated_text']))
            if generated_embedding:
                data['generated_embedding'] = generated_embedding
            
        if 'converted_text' in data and 'converted_embedding' not in data:
            converted_embedding = embed_text(truncate_text(data['converted_text']))
            if converted_embedding:
                data['converted_embedding'] = converted_embedding
        
        return data if ('generated_embedding' in data or 'converted_embedding' in data) else None
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_directory(input_dir: str, output_dir: str, max_workers: int = 4):
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
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Processing {len(json_files)} files...")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress bar
        results = list(tqdm(
            executor.map(lambda x: process_file(x, output_dir), json_files),
            total=len(json_files),
            desc="Generating embeddings"
        ))
    
    # Save processed files
    processed_count = 0
    for file_path, result in zip(json_files, results):
        if result:
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
    run_name = "taco_instruct_llama_8b_single_slerp_0.5/"
    input_directory = f"messi-thinking/{run_name}"
    # input_directory = f"/scratch/gpfs/vv7118/projects/messi-thinking/{run_name}"
    output_directory = f"embeddings/{run_name}"
    os.makedirs(output_directory, exist_ok=True)
    
    process_directory(
        input_directory,
        output_directory,
        max_workers=10  # Adjust based on your needs
    )
