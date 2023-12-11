import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    '''
    the collator class to collate the data for finetuning the whisper model
    '''

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, Any]:

        '''
        split inputs and labels since they have to be of different lengths and need different padding methods
        '''

        # first treat the audio inputs by simply returning torch tensors
        # batch = [{"input_features": feature['audio']['array']} for feature in features]

        raw_audio_array_list = [feature['audio']['array'] for feature in features]
        length_of_tensors = [elem.size(0) for elem in raw_audio_array_list]
        max_length_of_tensors = max(length_of_tensors)

        # padding audio to the same length
        audio_array_padded_list = [torch.nn.functional.pad(elem, pad=(0, max_length_of_tensors - elem.size(0)), value=0) for elem in raw_audio_array_list]

        # temp fix: convert the batch from torch.Tensor to numpy array due to the bug that torch.Tensor does not produce batch transcription
        audio_array_padded_list = [batch_elem.numpy() for batch_elem in audio_array_padded_list]

        batch= {}
        batch['audio_array'] = audio_array_padded_list
        batch['text'] = [feature['text'] for feature in features]

        return batch