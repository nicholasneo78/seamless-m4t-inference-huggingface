import librosa
from transformers import SeamlessM4TProcessor


def prepare_dataset(
    batch, 
    processor: SeamlessM4TProcessor,
    root_path_to_be_removed: str, 
    root_path_to_be_replaced: str,
):
        
    '''
    to prepare the final dataset that is to be fed into the pretrained whisper model for finetuning, this method is used in the huggingface dataset.map(...) call
    ---

    batch: the batch of data that would be processed
    processor: the processor object to do feature extraction and tokenzation of the dataset
    root_path_to_be_removed: the absolute path to be removed if there is a change in the data directory
    root_path_to_be_replaced: the absolute path to be replaced if there is a change in the data directory
    ---
    
    returns the batch of data after being processed
    '''

    # some filepath preprocessing
    batch['file'] = batch['file'].replace(root_path_to_be_removed, root_path_to_be_replaced)

    # retrieve the audio features from the filepath
    audio = batch["audio"]
    audio['path'] = audio['path'].replace(root_path_to_be_removed, root_path_to_be_replaced)
    audio["array"] = librosa.load(audio["path"], sr=audio["sampling_rate"])[0]

    # TODO: investigate the issue of loading the language of the audio from manifest file instead 
    # set the tokenizer's language based on the data entry
    processor.tokenizer.set_src_lang_special_tokens(
        batch['language'],
    )
    
    processor.tokenizer.set_tgt_lang_special_tokens(
        batch['language'],
    )
    
    return batch
