import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from .sub_llm_wrapper import ask_sub_model
from .sub_personality import SDF_ENCODING_INSTRUCTION, SHAPE_GENERATION_INSTRUCTION, URL_FROM_TEXT_INSTRUCTION, SONG_INFO_EXTRACTION_INSTRUCTION
from AI_Model.log import log
import json, re

def _extract_json(text: str) -> str:
    """
    Extract JSON from LLM output.
    Handles ```json ... ``` and raw JSON.
    """
    text = text.strip()

    # If wrapped in ```json ``` or ``` ```
    if text.startswith("```"):
        # Remove ```json or ``` and trailing ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()

def sub_parse_url(text: str) -> str:
    """
    Extracts URL from the given text using the sub-agent model.
    """
    messages = f"Extract the URL from the following text:\n\n'{text}'\n\nIf no URL is present, respond with 'No URL found.'"

    log(f"Parsing URL from text using sub-agent model...", "SUB AGENT")
    response = ask_sub_model(URL_FROM_TEXT_INSTRUCTION, messages)
    return response.strip()

def sub_extract_song_info(text: str) -> str:
    """
    Extracts song Name and Creator from the given text using the sub-agent model.
    """
    messages = f"Extract the Name and Creator of the song from the following text:\n\n'{text}'\n\nRespond in the format: 'Name:<Song Name> Creator:<Creator Name(s)>'"

    log(f"Extracting song info from text using sub-agent model...", "SUB AGENT")
    response = ask_sub_model(SONG_INFO_EXTRACTION_INSTRUCTION, messages)
    return response.strip()

# Add this function to your sub_agent.py file



def sub_generate_shape(description: str) -> dict:
    """
    Generates 3D shape encoding from text description using the sub-agent model.
    Returns a dictionary with geometric primitives.
    """
    messages = f"Generate 3D shape primitives for: '{description}'"
    
    log(f"Generating 3D shape for '{description}' using sub-agent model...", "SUB AGENT")
    response = ask_sub_model(SHAPE_GENERATION_INSTRUCTION, messages)
    
    try:
        # Parse the JSON response
        shape_data = json.loads(response.strip())
        log(f"Successfully generated {len(shape_data.get('primitives', []))} primitives", "SUB AGENT")
        return shape_data
    except json.JSONDecodeError as e:
        log(f"Failed to parse JSON response: {e}", "SUB AGENT")
        log(f"Raw response was: {response}", "SUB AGENT")
        # Return a default sphere if parsing fails
        return {
            "primitives": [
                {"type": "sphere", "center": [0, 0, 0], "radius": 1.0, "density": 1.0}
            ]
        }


def sub_generate_sdf_encoding(description: str, model: str = "qwen2.5:7b") -> list:
    """
    Generates SDF encoding vector from text description using the sub-agent model.
    Returns a list of 64 floating point numbers representing the shape.
    """
    messages = f"Generate SDF encoding for: '{description}'"
    
    log(f"Generating SDF encoding for '{description}' using {model}...", "SUB AGENT")
    response = ask_sub_model(SDF_ENCODING_INSTRUCTION, messages, model=model)
    
    try:
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        if response.startswith("json"):
            response = response[4:].strip()
        
        # Parse JSON
        data = json.loads(response)
        encoding = data.get("encoding", [])
        
        if len(encoding) == 64:
            log(f"Successfully generated SDF encoding with 64 values", "SUB AGENT")
            return encoding
        else:
            log(f"Invalid encoding length: {len(encoding)}, expected 64", "SUB AGENT")
            # Return random encoding as fallback
            import random
            return [random.uniform(-0.5, 0.5) for _ in range(64)]
            
    except json.JSONDecodeError as e:
        log(f"Failed to parse JSON response: {e}", "SUB AGENT")
        log(f"Raw response was: {response}", "SUB AGENT")
        # Return random encoding as fallback
        import random
        return [random.uniform(-0.5, 0.5) for _ in range(64)]