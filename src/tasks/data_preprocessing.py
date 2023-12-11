from typing import Tuple, List, Dict
import json

from modules import create_huggingface_manifest

class BuildHuggingFaceDataManifest:

    '''
    Generate updated manifest file of the form where the huggingface datasets library can take in
    '''

    def __init__(
        self,
        input_manifest_path: str,
        output_manifest_path: str,
    ) -> None:

        '''
        input_manifest_path: the path to retrieve the manifest file with the information of the audio path and annotations
        output_manifest_path: the file path where the huggingface manifest file will reside after preprocessing, the file would be in the json format
        '''

        self.input_manifest_path = input_manifest_path
        self.output_manifest_path = output_manifest_path

    
    def build_hf_data_manifest(self) -> Tuple[Dict[str, str], str]:

        '''
        main method to process the nemo format manifest file into the huggingface format manifest file
        '''

        # load all the data from the manifest
        data_list = create_huggingface_manifest(input_manifest_path=self.input_manifest_path)

        # form the final json manifest that is ready for export
        data_dict = {}
        data_dict['data'] = data_list

        # export to the final json format
        with open(f'{self.output_manifest_path}', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_dict, indent=2, ensure_ascii=False))

        # returns the final preprocessed dataframe and the filepath of the pickle file
        return data_dict, self.output_manifest_path
    

    def __call__(self):
        return self.build_hf_data_manifest()