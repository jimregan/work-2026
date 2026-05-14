
This is a fantastic project! Using the recently released **Gemma 4** family for a language-learning pipeline is a great use of its new multimodal and agentic capabilities.

Since you are using a local **Ollama** setup, you can take advantage of the fact that Gemma 4 now supports **native audio and vision** in its smaller "edge" variants ($E2B$ and $E4B$), while the larger models ($26B$ and $31B$) excel at the reasoning required for coordination and CEFR-level vocabulary filtering.

Here is a recommended architecture for your pipeline:

---

## 1. The Multi-Model Pipeline

To achieve the best results, you should leverage the specific strengths of each Gemma 4 variant.

### Model A: The "Ear" (Gemma-4-E4B)

- **Role:** Audio Transcription (ASR).
    
- **Why:** The $E2B$ and $E4B$ models are the only ones in the family with **native audio support**. $E4B$ (4.5B effective parameters) provides a better balance of accuracy for Hungarian phonetics than the smaller $E2B$.
    
- **Task:** You will pass the audio stream directly to the model. It supports Hungarian out-of-the-box.
    

### Model B: The "Eye" (Gemma-4-31B or E4B)

- **Role:** OCR & Visual Context.
    
- **Why:** Reels often have "burned-in" captions or text overlays that clarify the audio.
    
- **Task:** Extract frames from the Reel (e.g., 1 frame per second) and pass them to the model to transcribe on-screen text.
    

### Model C: The "Brain" (Gemma-4-31B-it)

- **Role:** Coordination, Translation, and Vocabulary Filtering.
    
- **Why:** This is the heavyweight dense model. It has the "reasoning" power to compare the audio transcript vs. the OCR text, pick the most likely correct words, and perform the A2-level filtering.
    

---

## 2. Implementation Strategy

### Step 1: Pre-processing (External Tools)

Ollama handles model inference, but you’ll need a small Python script to prep the Reel:

1. **Extract Audio:** Use `ffmpeg` to pull a `.wav` file from the Reel.
    
2. **Extract Frames:** Use `opencv` to grab keyframes where text is visible.
    

### Step 2: The Ollama Workflow

You can orchestrate this using a simple Python script with the `ollama` library:

Python

```
import ollama

# 1. Get Audio Transcript (Hungarian)
audio_res = ollama.generate(
    model='gemma4:e4b',
    prompt='Transcribe this Hungarian audio accurately.',
    images=['./audio_as_input.wav'] # Note: Ensure your Ollama version supports the new audio tag
)

# 2. Get OCR Text
ocr_res = ollama.generate(
    model='gemma4:e4b',
    prompt='List all text visible in these frames.',
    images=['frame1.jpg', 'frame2.jpg']
)

# 3. Coordinate & Translate (The "Brain")
final_output = ollama.chat(
    model='gemma4:31b',
    messages=[{
        'role': 'user',
        'content': f"""
        Audio Transcript: {audio_res['response']}
        OCR Text: {ocr_res['response']}
        
        Task:
        1. Create a synchronized line-by-line Hungarian to English translation.
        2. Create a vocabulary list. 
        3. CRITICAL: Only include words at an A2 level or higher. Do NOT include basic A1 words (like 'szia', 'van', 'alma').
        """
    }]
)
```

---

## 3. Key Tips for Hungarian & A2 Filtering

- **Agglutination Handling:** Hungarian is an agglutinative language (prefixes/suffixes change the word significantly). When asking the "Brain" model for vocabulary, specifically instruct it to **"provide the dictionary (lemma) form of the word."**
    
- **Thinking Mode:** For the coordination step, enable Gemma 4’s new **"Thinking Mode"** by adding `<|think|>` to your system prompt. This allows the model to "reason" through why a word might be A2 vs. A1 before it gives you the final list.
    
- **Hardware Note:** * **Gemma-4-E4B** runs comfortably on 8GB VRAM.
    
    - **Gemma-4-31B** (quantized) usually requires 20GB-24GB VRAM. If you are tight on memory, use the **Gemma-4-26B-A4B** (the Mixture-of-Experts model), which is much faster and lighter on RAM.
        

### Recommended System Prompt for the "Brain":

> "You are a linguistic expert in Hungarian. Compare the provided audio and OCR data to create a perfect transcript. Translate it into natural English. Finally, extract vocabulary that a learner would encounter at the A2 level, specifically excluding the top 500 most common A1 Hungarian words."

Would you like help with a specific Python script to handle the `ffmpeg` extraction part of this pipeline?