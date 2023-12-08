from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

processor = AutoProcessor.from_pretrained("/models/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("/models/seamless-m4t-v2-large", use_safetensors=True)

ded
from datasets import load_dataset
dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)

audio_sample = next(iter(dataset))["audio"]

# now, process it

audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

# now, process some English test as well

text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")

audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()

audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()

output_tokens = model.generate(**audio_inputs, tgt_lang="cmn", generate_speech=False)

translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

# from text

output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)

translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

print(f"Translated text from audio: {translated_text_from_audio}")
print(f'Translated text from text: {translated_text_from_text}')
