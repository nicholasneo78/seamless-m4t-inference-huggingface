import os
from typing import List, Dict
from modules import WER, CER, MER, load_manifest_nemo
import logging

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

class GetMetricsFromJSON:
    
    '''
    to get the audio metrics from the JSON file with key "prediction" and "ground truth" after running the evaluate_model.py script to generate the json file
    '''

    def __init__(self, cfg: Dict, is_remote: bool) -> None:
    
        '''
        input_json_dir (str): the json directory that was generated from evaluate_model.py
        '''

        if not is_remote:
            self.input_json_dir = cfg.get_metrics_from_json.input_json_dir
        else:
            self.input_json_dir = os.path.join(cfg.temp.dataset_json_path, cfg.get_metrics_from_json.manifest.manifest_path)
        

    def get_metrics_result(self) -> None:

        '''
        main method to print the WER and CER
        '''

        # load the dict list that was loaded from the json manifest file
        data = load_manifest_nemo(input_manifest_path=self.input_json_dir)

        # get the pred and the ground truth list
        pred_list = [pred['pred_str'] for pred in data]
        ground_truth_list = [ref['text'] for ref in data]

        # compute the WER
        get_wer = WER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_cer = CER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_mer = MER(
            predictions=pred_list,
            references=ground_truth_list
        )

        print()
        logging.getLogger('INFO').info("Test WER: {:.5f}".format(get_wer.compute()))
        logging.getLogger('INFO').info("Test CER: {:.5f}".format(get_cer.compute()))
        logging.getLogger('INFO').info("Test MER: {:.5f}".format(get_mer.compute()))
        logging.getLogger('INFO').info("Word Acc: {:.5f}\n".format(1-get_mer.compute()))


    def __call__(self):
        return self.get_metrics_result()