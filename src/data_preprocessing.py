import hydra
from tasks import BuildHuggingFaceDataManifest

# python3 data_preprocessing.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to convert the nemo manifest to the huggingface manifest 
    '''

    if cfg.data_preprocessing.do_split:

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=cfg.data_preprocessing.split.input_manifest_path_train,
            output_manifest_path=cfg.data_preprocessing.split.output_manifest_path_train,
        )()

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=cfg.data_preprocessing.split.input_manifest_path_dev,
            output_manifest_path=cfg.data_preprocessing.split.output_manifest_path_dev,
        )()

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=cfg.data_preprocessing.split.input_manifest_path_test,
            output_manifest_path=cfg.data_preprocessing.split.output_manifest_path_test,
        )()

    else:

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=cfg.data_preprocessing.no_split.input_manifest_path,
            output_manifest_path=cfg.data_preprocessing.no_split.output_manifest_path,
        )()

if __name__ == '__main__':
    main()