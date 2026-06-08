import os
import glob
import json
import re
import time
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict

POLICY_PROMPT = (
    "You are a driving-safety analyst. The image is a grid of sequential dashcam\n"
    "frames from ONE continuous clip, ordered left-to-right, top-to-bottom, showing\n"
    "a single developing driving situation from the ego car whose dashcam recorded it.\n\n"
    "Read the frames in order, identify the hazard building up, and output a JSON\n"
    "object with exactly these three fields:\n\n"
    "  \"trigger\":      The specific observable scene conditions that create danger\n"
    "                   (road type, weather, other vehicle behavior, visibility).\n"
    "                   One concise sentence. Must be matchable against a live scene\n"
    "                   description without knowing the outcome.\n\n"
    "  \"latent_risk\":  What could go wrong mechanically and why — the failure mode\n"
    "                   that connects the trigger to a collision. One concise sentence.\n\n"
    "  \"mitigation\":   The specific corrective action the ego driver should take to\n"
    "                   prevent the crash. One concise sentence.\n\n"
    "Rules:\n"
    "- Each field must be independently meaningful and self-contained.\n"
    "- Do NOT merge fields into one sentence.\n"
    "- Make all three fields GENERALIZABLE to any driver in a similar situation,\n"
    "  not a description of this specific clip.\n"
    "- Base everything solely on what is visible. Do not invent details.\n"
    "- Output only the raw JSON object. No markdown, no explanation, no extra text.\n"
)

REQUIRED_TRIPLET_KEYS = {"trigger", "latent_risk", "mitigation"}


def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output, stripping markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON from model output:\n{text}")

def create_frame_grid_from_images(image_paths, grid_size=(3, 3)):
    """Creates a grid image from a list of image paths."""
    frames = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            frames.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    num_frames_needed = grid_size[0] * grid_size[1]
    
    if not frames:
        raise ValueError("Could not load any frames.")
        
    if len(frames) < num_frames_needed:
        while len(frames) < num_frames_needed:
            frames.append(frames[-1].copy())
            
    # Subsample to get exactly num_frames_needed if we have more
    if len(frames) > num_frames_needed:
        step = max(1, len(frames) // num_frames_needed)
        selected_frames = []
        for i in range(num_frames_needed):
            selected_frames.append(frames[min(i * step, len(frames) - 1)])
        frames = selected_frames

    # Create grid
    w, h = frames[0].size
    grid_img = Image.new('RGB', (w * grid_size[0], h * grid_size[1]))
    
    for i, frame in enumerate(frames):
        if frame.size != (w, h):
            frame = frame.resize((w, h))
        row = i // grid_size[0]
        col = i % grid_size[0]
        grid_img.paste(frame, (col * w, row * h))
        
    return grid_img

def main():
    # Configuration
    OUTPUT_FILE = "crash_policies.jsonl"
    MODEL_NAME = 'gemini-3.1-flash-lite'
    MAX_RETRIES = 3
    RETRY_DELAY = 30 # seconds to wait if rate limited
    
    # Load environment variables
    load_dotenv(override=True)
    api_keys = [
        os.getenv("GEM_KEY0"),
        os.getenv("GEM_KEY1"),
        os.getenv("GEM_KEY2")
    ]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        print("Error: No GEM_KEYs found in .env")
        return
        
    print(f"{len(api_keys)} API Keys loaded successfully.")
    
    current_key_idx = 0
    
    # Configure Gemini
    genai.configure(api_key=api_keys[current_key_idx])
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Load already processed clips to allow resuming
    processed_clips = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'clip_id' in data:
                        processed_clips.add(data['clip_id'])
                except:
                    pass
    print(f"Found {len(processed_clips)} already processed clips in {OUTPUT_FILE}.")
    
    # Find dataset
    import kagglehub
    try:
        print("Locating dataset (downloading if necessary, this may take a moment)...")
        dataset_path = kagglehub.dataset_download('asefjamilajwad/car-crash-dataset-ccd')
        
        image_files = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
        if not image_files:
            print("Error: Could not find any .jpg files in the dataset.")
            return
            
        # Group by prefix (assuming format C_001262_15.jpg where 15 is frame)
        clip_groups = defaultdict(list)
        for img_path in image_files:
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 3:
                prefix = "_".join(parts[:-1]) # e.g. C_001262
                try:
                    frame_num = int(parts[-1].split('.')[0])
                    clip_groups[prefix].append((frame_num, img_path))
                except ValueError:
                    pass
                    
        if not clip_groups:
            print("Error: Could not parse clip groups from filenames.")
            return
            
        print(f"Total unique clips found: {len(clip_groups)}")
        
    except Exception as e:
        print(f"Error accessing dataset: {e}")
        return

    # Process clips
    with open(OUTPUT_FILE, 'a') as f:
        for clip_id, frames_data in clip_groups.items():
            if clip_id in processed_clips:
                continue
                
            print(f"Processing clip {clip_id}...")
            
            # Sort frames by frame number
            clip_frames = sorted(frames_data, key=lambda x: x[0])
            frame_paths = [path for _, path in clip_frames]
            
            # Build grid
            grid_image = create_frame_grid_from_images(frame_paths)
            
            # Call Gemini API with retry logic
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    response = model.generate_content([POLICY_PROMPT, grid_image])
                    raw_text = response.text.strip()

                    # Parse structured JSON triplet
                    try:
                        parsed = extract_json(raw_text)
                        missing = REQUIRED_TRIPLET_KEYS - set(parsed.keys())
                        if missing:
                            raise ValueError(f"Missing required keys: {missing}")
                        record = {
                            "clip_id": clip_id,
                            "trigger": parsed["trigger"],
                            "latent_risk": parsed["latent_risk"],
                            "mitigation": parsed["mitigation"],
                        }
                        print(f"  -> Success: trigger={parsed['trigger'][:80]}...")
                    except (ValueError, KeyError) as parse_err:
                        record = {"clip_id": clip_id, "error": "parse_failed", "raw": raw_text}
                        print(f"  -> Parse failed for {clip_id}: {parse_err}")

                    f.write(json.dumps(record) + '\n')
                    f.flush()
                    success = True
                    time.sleep(1) # Small delay to be polite
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "Quota Exceeded" in error_str:
                        current_key_idx = (current_key_idx + 1) % len(api_keys)
                        print(f"  -> Rate limited (429/Quota). Switching to API key index {current_key_idx}... (Attempt {attempt+1}/{MAX_RETRIES})")
                        genai.configure(api_key=api_keys[current_key_idx])
                        model = genai.GenerativeModel(MODEL_NAME)
                        time.sleep(2)
                    else:
                        print(f"  -> API Error on {clip_id}: {e}")
                        time.sleep(5) # Small wait on other errors before retrying
                        
            if not success:
                print(f"Failed to process {clip_id} after {MAX_RETRIES} attempts. Skipping.")

if __name__ == "__main__":
    main()
