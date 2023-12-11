import datasets
from jiwer import compute_measures, cer, mer
from transformers import WhisperProcessor
from typing import Dict, List
from tqdm import tqdm
from modules import TextPostProcessingManager
import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class WER(datasets.Metric):

    '''
    WER metrics
    '''

    def __init__(self, predictions=None, references=None, concatenate_texts=False):
        self.predictions = predictions
        self.references = references
        self.concatenate_texts = concatenate_texts

    def compute(self):
        if self.concatenate_texts:
            return compute_measures(self.references, self.predictions)['wer']
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(self.predictions, self.references):
                measures = compute_measures(reference, prediction)
                incorrect += measures['substitutions'] + measures['deletions'] + measures['insertions']
                total += measures['substitutions'] + measures['deletions'] + measures['hits']
            return incorrect/total
        

class CER:
    
    '''
    CER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return cer(reference=self.references, hypothesis=self.predictions)


class MER:
    
    '''
    MER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return mer(reference=self.references, hypothesis=self.predictions)


class GetF1Score:

    '''
    The F1-score computation for the zero-shot LID prediction
    '''

    def __init__(self, predictions=None, references=None, language=None):
        self.predictions = predictions
        self.references = references
        self.language = language

    def compute(self):
        return classification_report(y_true=self.references, y_pred=self.predictions, labels=self.language)
    

class GetConfusionMatrix:

    '''
    The confusion matrix computation for the zero-shot LID prediction
    '''

    def __init__(self, predictions=None, references=None, language=None):
        self.predictions = predictions
        self.references = references
        self.language = language

    def compute(self):

        logging.getLogger('INFO').info('Predicted: Vertical Axis; Reference: Horizontal Axis')
        logging.getLogger('INFO').info(f'Order of the languages: {self.language}')

        return confusion_matrix(y_true=self.references, y_pred=self.predictions, labels=self.language)
    

def compute_metrics(
    pred, 
    processor: WhisperProcessor,
    data_label: str,
    language_list: List[str]
) -> Dict[str, float]:
    
    '''
    to evaluate the wer of the model on the validation set during finetuning
    ---

    pred: predicted transcription from the validation set
    processor: the processor object to do feature extraction and tokenzation of the dataset
    data_label: a flag to the data in the case if there is a text postprocessing step unique to this data
    finetuned_language_dict: to load the language for the model's decoder to decode the speech in the desired language
    ---
    returns a dictionary with the value as the WER value
    '''
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # debug to see the tokens in the labels and the predictions
    print(pred_ids[0])
    print()
    print()
    print(label_ids[0])
    print()

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # do the text preprocessing here to normalise the predicted text form whisper
    logging.getLogger('INFO').info('Post processing the text now...')

    pred_str_processed = [
        TextPostProcessingManager(
            label=data_label,
            language=language,
        ).process_data(
            text=sentence
        ) for sentence, language in tqdm(zip(pred_str, language_list))
    ]

    label_str_processed = [
        TextPostProcessingManager(
            label=data_label,
            language=language
        ).process_data(
            text=sentence
        ) for sentence, language in tqdm(zip(label_str, language_list))
    ]

    logging.getLogger('INFO').info('Post processing of text done!')

    # for debugging purpose
    logging.getLogger('INFO').info(f'Pred str: {pred_str_processed[:3]}\n\n')
    logging.getLogger('INFO').info(f'Label str: {label_str_processed[:3]}\n')

    get_wer = WER(predictions=pred_str_processed, references=label_str_processed)

    return {"wer": get_wer.compute()}


def get_confidence_score(batch_token_scores: List[np.array]):

        '''
        takes in a list of batched numpy arrays and outputs the confidence scores of this batch of data

        https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        
        ---
        batch_token_scores: the numpy array of the token scores in a defined batch number
        ---
        returns the confidence scores of the prediction in a batch
        '''

        # instantiate the list to get the confidence score
        confidence_score_list = []

        # get the confidence score
        for batch_token_score in batch_token_scores:

            a = batch_token_score
            e_x = np.exp(a - np.reshape(np.max(a, axis=1), (-1, 1)))
            softmax = np.divide(e_x, np.reshape(e_x.sum(axis=1), (-1, 1)))
            confidence_score = np.max(softmax, axis=1)
            confidence_score_list.append(confidence_score)

        # convert to numpy array and average the confidence score of the utterance
        avg_confidence_score_array = np.mean(np.asarray(confidence_score_list), axis=0)

        # debugging purpose
        # logging.getLogger('INFO').info(avg_confidence_score_array)

        return avg_confidence_score_array