from transformers import SeamlessM4Tv2Model, SeamlessM4TProcessor
import torch
from tqdm import tqdm
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = SeamlessM4TProcessor.from_pretrained("/models/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("/models/seamless-m4t-v2-large", use_safetensors=True).to(device)

from modules import TextPostProcessingManager, WER, CER, MER, load_huggingface_manifest_evaluation, get_confidence_score, extract_file_path_from_json, extract_duration_from_json, prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding

# Setup logging in a nice readable format
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

# PARAMS
test_dir = '/datasets/fleurs-en/test_m4t.json'
dev_dir = '/datasets/fleurs-en/dev_m4t.json'

# instantiate the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# load the test manifest files and form the IterableDatasetDict
dataset = load_huggingface_manifest_evaluation(
    test_dir=test_dir
)

# process the dataset to get the ground truth labels and the input features
dataset = dataset.map(
    lambda x: prepare_dataset(
        x, 
        processor=processor,
        root_path_to_be_removed='',
        root_path_to_be_replaced='',
    ),
).with_format('torch')

print(next(iter(dataset['test']))['audio']['array'])
print()
print()
print(next(iter(dataset['test'])))
print()

eval_dataloader = torch.utils.data.DataLoader(
    dataset["test"],
    batch_size=4,
    collate_fn=data_collator
)

model.eval()

for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            # print(batch)
            # print(batch[0].shape)
            audio_inputs = processor(audios=batch['audio_array'], return_tensors="pt", src_lang='eng', sampling_rate=16000).to(device)
            text_inputs = processor(text=batch['text'], return_tensors="pt", src_lang='eng').to(device)
            # print()
            # print()
            # # print(audio_inputs.input_features)
            # # print(len(audio_inputs.input_features))
            # print()
            # print()
            output_tokens_s2t = model.generate(**audio_inputs, tgt_lang='eng', generate_speech=False)
            output_tokens_t2t = model.generate(**text_inputs, tgt_lang='cmn', generate_speech=False)

            translated_text_from_audio = processor.batch_decode(output_tokens_s2t[0].tolist(), skip_special_tokens=True)
            translated_text_from_text = processor.batch_decode(output_tokens_t2t[0].tolist(), skip_special_tokens=True)
            print()
            print(f'Ground Truth text: {batch["text"]}')
            print(f'Translated text from audio: {translated_text_from_audio}')
            print(f'Translated text from text: {translated_text_from_text}')
            print()
            
