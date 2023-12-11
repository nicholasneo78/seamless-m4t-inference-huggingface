from transformers import SeamlessM4Tv2Model, SeamlessM4TProcessor
import torch
from tqdm import tqdm
import logging
from typing import Dict, List
import os
import json

from modules import TextPostProcessingManager, WER, CER, MER, load_huggingface_manifest_evaluation, extract_file_path_from_json, extract_duration_from_json, prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding

# Setup logging in a nice readable format
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class M4TEvaluation:

    '''
    evaluation of the zero-shot model
    '''

    def __init__(
        self, 
        cfg: Dict,
        is_remote: bool
    ) -> None:
        
        '''
        test_dir (str): path to the test manifest file in the huggingface format
        root_path_to_be_removed (str): the original old root path to be replaced in the json manifest file
        root_path_to_be_replaced (str): the new root path to replace the old one in the json manifest file
        processor_path (str): directory of the processor path after finetuning
        input_model_path (str): directory of the input model path, can be online or offline 
        data_label (str): the name of the dataset used in the evaluation (can be any string)
        output_pred_dir (str): json file that contains the filename, predictions, ground truth and the confidence scores
        data_loader_batch_size (int): the size of the torch dataloader for evaluation
        skip_special_tokens (bool): to skip or not to skip the special tokens when model.generate is called 
        is_s2t (bool): boolean to check whether speech to text translation is performed
        is t2t (bool): boolean to check whether text to text translation is performed
        input_language_s2t (str): language of the input speech for s2t task
        input_language_t2t (str): language of the input speech for t2t task
        output_language_s2t (str): language of the output text for s2t task
        output_language_t2t (str): language of the output text for t2t task
        '''

        # local
        if not is_remote:
            self.test_dir = cfg.evaluate_model.data.manifest_path_test
            self.root_path_to_be_removed = cfg.evaluate_model.data.root_path_to_be_removed
            self.root_path_to_be_replaced = cfg.evaluate_model.data.root_path_to_be_replaced
            self.data_label = cfg.evaluate_model.data.data_label
            self.processor_path = cfg.evaluate_model.model.processor_path
            self.input_model_path = cfg.evaluate_model.model.input_model_path
            self.output_pred_dir_s2t = cfg.evaluate_model.model.s2t.output_pred_dir
            self.output_pred_dir_t2t = cfg.evaluate_model.model.t2t.output_pred_dir
            self.data_loader_batch_size = cfg.evaluate_model.model.data_loader_batch_size
            self.generate_speech = cfg.evaluate_model.model.generate_speech
            self.skip_special_tokens = cfg.evaluate_model.model.skip_special_tokens
            self.is_s2t = cfg.evaluate_model.model.is_s2t
            self.is_t2t = cfg.evaluate_model.model.is_t2t
            self.input_language_s2t = cfg.evaluate_model.model.s2t.input_language
            self.output_language_s2t = cfg.evaluate_model.model.s2t.output_language
            self.input_language_t2t = cfg.evaluate_model.model.t2t.input_language
            self.output_language_t2t = cfg.evaluate_model.model.t2t.output_language

        # TODO: clearml implementation
        else:
            pass
            
        self.SAMPLING_RATE = 16000
        self.model = SeamlessM4Tv2Model.from_pretrained(self.input_model_path, use_safetensors=True)
        self.processor = SeamlessM4TProcessor.from_pretrained(self.processor_path)

    
    def m4t_evaluation(self) -> None:

        '''
        main method to do the evaluation and calculate the WER and json file that contains the prediction, ground truth and the filename
        '''

        if not (self.is_s2t or self.is_t2t):
            raise Exception(f"At least one of task (self.is_s2t or self.is_t2t or both) has to set to true")

        # instantiate the data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        # take in the test manifest files and form the IterableDatasetDict
        dataset = load_huggingface_manifest_evaluation(
            test_dir=self.test_dir
        )

        # process the dataset to get the ground truth labels and the input features
        dataset = dataset.map(
            lambda x: prepare_dataset(
                x, 
                processor=self.processor,
                root_path_to_be_removed='',
                root_path_to_be_replaced='',
            ),
        ).with_format('torch')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.model.to(device)

        # set up dataloader for the test set
        eval_dataloader = torch.utils.data.DataLoader(
            dataset["test"],
            batch_size=self.data_loader_batch_size,
            collate_fn=data_collator
        )

        if self.is_s2t:
            s2t_dict = {
                'decoded_preds_list_raw': [],
                'decoded_labels_list_raw': [],
                'decoded_preds_list_normalized': [],
                'decoded_labels_list_normalized': [],
            }

        if self.is_t2t:
            t2t_dict = {
                'decoded_preds_list_raw': [],
                'decoded_labels_list_raw': [],
                'decoded_preds_list_normalized': [],
                'decoded_labels_list_normalized': [],
            }

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    if self.is_s2t:
                        audio_inputs = self.processor(audios=batch['audio_array'], return_tensors="pt", src_lang=self.input_language_s2t, sampling_rate=self.SAMPLING_RATE).to(device)
                        output_tokens_s2t = model.generate(**audio_inputs, tgt_lang=self.output_language_s2t, generate_speech=self.generate_speech)
                        translated_text_from_audio_raw = self.processor.batch_decode(output_tokens_s2t[0].tolist(), skip_special_tokens=self.skip_special_tokens)
                        ground_truth_text_s2t_raw = batch['text']

                        translated_text_from_audio_normalized = [
                            TextPostProcessingManager(
                            label=self.data_label,
                            language=self.output_language_s2t,
                            ).process_data(
                                text
                            ) for text in translated_text_from_audio_raw
                        ]

                        ground_truth_text_s2t_normalized = [
                            TextPostProcessingManager(
                            label=self.data_label,
                            language=self.output_language_s2t,
                            ).process_data(
                                text
                            ) for text in ground_truth_text_s2t_raw
                        ]

                        logging.getLogger('INFO').info(f'Predicted (raw): {translated_text_from_audio_raw}\n')
                        logging.getLogger('INFO').info(f'Ground Truth (raw): {ground_truth_text_s2t_raw}\n\n')

                        logging.getLogger('INFO').info(f'Predicted (normalized): {translated_text_from_audio_normalized}\n')
                        logging.getLogger('INFO').info(f'Ground Truth (normalized): {ground_truth_text_s2t_normalized}\n\n')

                        s2t_dict['decoded_preds_list_raw'].extend(translated_text_from_audio_raw)
                        s2t_dict['decoded_labels_list_raw'].extend(ground_truth_text_s2t_raw)
                        s2t_dict['decoded_preds_list_normalized'].extend(translated_text_from_audio_normalized)
                        s2t_dict['decoded_labels_list_normalized'].extend(ground_truth_text_s2t_normalized)

                    if self.is_t2t:
                        text_inputs = self.processor(text=batch['text'], return_tensors="pt", src_lang=self.input_language_t2t).to(device)
                        output_tokens_t2t = model.generate(**text_inputs, tgt_lang=self.output_language_t2t, generate_speech=self.generate_speech)
                        translated_text_from_text_raw = self.processor.batch_decode(output_tokens_t2t[0].tolist(), skip_special_tokens=self.skip_special_tokens)
                        ground_truth_text_t2t_raw = batch['text']

                        translated_text_from_text_normalized = [
                            TextPostProcessingManager(
                            label=self.data_label,
                            language=self.output_language_t2t,
                            ).process_data(
                                text
                            ) for text in translated_text_from_text_raw
                        ]

                        ground_truth_text_t2t_normalized = [
                            TextPostProcessingManager(
                            label=self.data_label,
                            language=self.input_language_t2t,
                            ).process_data(
                                text
                            ) for text in ground_truth_text_t2t_raw
                        ]

                        logging.getLogger('INFO').info(f'Predicted (raw): {translated_text_from_text_raw}\n')
                        logging.getLogger('INFO').info(f'Ground Truth (raw): {ground_truth_text_t2t_raw}\n\n')

                        logging.getLogger('INFO').info(f'Predicted (normalized): {translated_text_from_text_normalized}\n')
                        logging.getLogger('INFO').info(f'Ground Truth (normalized): {ground_truth_text_t2t_normalized}\n\n')

                        t2t_dict['decoded_preds_list_raw'].extend(translated_text_from_text_raw)
                        t2t_dict['decoded_labels_list_raw'].extend(ground_truth_text_t2t_raw)
                        t2t_dict['decoded_preds_list_normalized'].extend(translated_text_from_text_normalized)
                        t2t_dict['decoded_labels_list_normalized'].extend(ground_truth_text_t2t_normalized)

        # Calculate WER. CER, MER IF self.is_s2t is checked
        if self.is_s2t:
            get_wer = WER(
                predictions=s2t_dict['decoded_preds_list_normalized'],
                references=s2t_dict['decoded_labels_list_normalized']
            )

            get_cer = CER(
                predictions=s2t_dict['decoded_preds_list_normalized'],
                references=s2t_dict['decoded_labels_list_normalized']
            )

            get_mer = MER(
                predictions=s2t_dict['decoded_preds_list_normalized'],
                references=s2t_dict['decoded_labels_list_normalized']
            )

            print()
            logging.getLogger('INFO').info(f"Test WER: {get_wer.compute():.5f} | Data Label: {self.data_label}")
            logging.getLogger('INFO').info(f"Test CER: {get_cer.compute():.5f} | Data Label: {self.data_label}")
            logging.getLogger('INFO').info(f"Test MER: {get_mer.compute():.5f} | Data Label: {self.data_label}")
            logging.getLogger('INFO').info(f"Word Acc: {(1-get_mer.compute()):.5f} | Data Label: {self.data_label}\n")


        # extracts the list of test set audio directories
        audio_filepath_list = extract_file_path_from_json(test_dir=self.test_dir)
        duration_list = extract_duration_from_json(test_dir=self.test_dir)

        logging.getLogger('INFO').info('Writing the predictions and labels to a json file...')

        # for s2t
        if self.is_s2t:
            os.remove(self.output_pred_dir_s2t) if os.path.exists(self.output_pred_dir_s2t) else None
            with open(self.output_pred_dir_s2t, 'w+', encoding='utf-8') as f:
                for audio_filepath, pred_raw, ref_raw, pred_norm, ref_norm, duration in tqdm(
                    zip(
                        audio_filepath_list,
                        s2t_dict['decoded_preds_list_raw'],
                        s2t_dict['decoded_labels_list_raw'],
                        s2t_dict['decoded_preds_list_normalized'],
                        s2t_dict['decoded_labels_list_normalized'],
                        duration_list
                    )
                ):
                    f.write(
                        json.dumps(
                            {
                                'audio_filepath': audio_filepath,
                                'text': ref_norm,
                                'pred_str': pred_norm,
                                'text_raw': ref_raw,
                                'pred_str_raw': pred_raw,
                                'duration': duration,
                            },
                            ensure_ascii=False
                        ) + '\n'
                    )
        
        if self.is_t2t:
            os.remove(self.output_pred_dir_t2t) if os.path.exists(self.output_pred_dir_t2t) else None
            with open(self.output_pred_dir_t2t, 'w+', encoding='utf-8') as f:
                for audio_filepath, pred_raw, ref_raw, pred_norm, ref_norm, duration in tqdm(
                    zip(
                        audio_filepath_list,
                        t2t_dict['decoded_preds_list_raw'],
                        t2t_dict['decoded_labels_list_raw'],
                        t2t_dict['decoded_preds_list_normalized'],
                        t2t_dict['decoded_labels_list_normalized'],
                        duration_list
                    )
                ):
                    f.write(
                        json.dumps(
                            {
                                'audio_filepath': audio_filepath,
                                'text': ref_norm,
                                'pred_str': pred_norm,
                                'text_raw': ref_raw,
                                'pred_str_raw': pred_raw,
                                'duration': duration,
                            },
                            ensure_ascii=False
                        ) + '\n'
                    )

    def __call__(self):
        return self.m4t_evaluation()