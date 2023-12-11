import os
import csv
import json
import librosa
from tqdm import tqdm
from typing import Dict, List

class TSVToJsonManifest:

    '''
    converts the transcription in the tsv format into the general manifest json file
    '''

    def __init__(
            self,
            root_folder: str,
            input_tsv: str,
            output_json: str,
            audio_subdir: str,
            language: str
    ) -> None:
        
        self.root_folder = root_folder
        self.input_tsv = input_tsv
        self.output_json = output_json
        self.audio_subdir = audio_subdir
        self.language = language

    
    def read_annotation(self) -> List[Dict[str, str]]:

        '''
        reads the annotations from the tsv files

        ---
        returns: a list of dict
        '''

        data_list = []

        # open the tsv file
        with open(self.input_tsv, 'r+') as f:
            data = csv.reader(f, delimiter='\t')

            for entry in data:
                
                if len(entry[3].split('\t')) != 1:
                    annotation = entry[3].split('\t')[0]
                else:
                    annotation = entry[3]

                data_dict = {
                    'filename': os.path.join(self.audio_subdir, entry[1]),
                    'annotation': annotation
                }

                data_list.append(data_dict)

        return data_list
    

    def export_json(self) -> None:

        '''
        to read the list of dictionary and then export them into a json file
        '''

        manifest_dict_list = []

        data_list = self.read_annotation()

        # iterate the list
        for data in tqdm(data_list):
            manifest_dict_list.append(
                {
                    'audio_filepath': data['filename'],
                    'duration': librosa.get_duration(path=os.path.join(self.root_folder, data['filename'])),
                    'language': self.language,
                    'text': data['annotation']
                }
            )

        # export to a json file
        with open(os.path.join(self.root_folder, self.output_json), 'w+') as f:
            for data_ in manifest_dict_list:
                f.write(json.dumps(data_) + '\n')
            

if __name__ == "__main__":

    t = TSVToJsonManifest(
        root_folder='/datasets/fleurs-en/',
        input_tsv='/datasets/fleurs-en/test.tsv',
        output_json='test_manifest_m4t.json',
        audio_subdir='test',
        language='eng'
    ).export_json()

    t = TSVToJsonManifest(
        root_folder='/datasets/fleurs-en/',
        input_tsv='/datasets/fleurs-en/train.tsv',
        output_json='train_manifest_m4t.json',
        audio_subdir='train',
        language='eng'
    ).export_json()

    t = TSVToJsonManifest(
        root_folder='/datasets/fleurs-en/',
        input_tsv='/datasets/fleurs-en/dev.tsv',
        output_json='dev_manifest_m4t.json',
        audio_subdir='dev',
        language='eng'
    ).export_json()

