import hydra

# python3 evaluate_model.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to do the model evaluation
    '''

    from tasks import M4TEvaluation

    _ = M4TEvaluation(
        cfg=cfg,
        is_remote=False
    )()


if __name__ == '__main__':
    main()