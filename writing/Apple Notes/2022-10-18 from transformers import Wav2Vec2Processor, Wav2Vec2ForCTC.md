from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
 from datasets import load_dataset
 import torch
 
 *# load model and processor*
 processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
 model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
     
 *# load dummy dataset and read soundfiles*
 ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 
 *# tokenize*
 input_values = processor(ds\[0\]\["audio"\]\["array"\], return_tensors="pt").input_values
 
 *# retrieve logits*
 with torch.no_grad():
   logits = model(input_values).logits
 
 *# take argmax and decode*
 predicted_ids = torch.argmax(logits, dim=-1)
 transcription = processor.batch_decode(predicted_ids)