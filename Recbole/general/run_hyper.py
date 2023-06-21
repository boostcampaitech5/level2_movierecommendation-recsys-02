from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import os
import argparse
import math
import sys

import logging
from logging import getLogger

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
)

def custom_objective_function(config_dict=None, config_file_list=None, saved=True):

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    # logger.info(config)
    
    '''
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    '''
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    logger.info(model)
    
    trainer = get_trainer(config["MODEL_TYPE"], model_name)(config, model)
    
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved, show_progress=True
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=True)

    ongoing_file = os.path.join(config['ongoing_file'])
    
    with open(ongoing_file, 'a') as fp:
        fp.write(
            "[model: "
            + str(model_name)
            + "]     Parameters "
            + str(config_dict)
            + "     Valid score: "
            + str(best_valid_score)
            + "\n\n"
        )
        
    # tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

def custom_export_result(hp, output_file=None):
        
        with open(output_file, "w") as fp:
            
            fp.write(
                "Best Parameters\n"
                + str(hp.best_params)
                + "\n\n"
                + "Best valid score\n"
                + str(hp.best_score)
                + "\n\n"
                +"--------------------------total results---------------------------\n\n"
            )
            
            for param, score in zip(hp.params_list, hp.score_list):
                fp.write(
                    "Parameters: "
                    + str(param)
                    + "     Valid score: "
                    + str(score)
                    + "\n\n"
                )

def hyperopt_tune(args):
    
    hp = HyperTuning(
        custom_objective_function,
        algo=args.algo, # exhaustive, bayes, random
        early_stop=10,
        max_evals=500,
        params_file=args.params_file,
        fixed_config_file_list=[args.config_file],
        display_file=args.display_file
    )
    
    hp.run()
    custom_export_result(hp, output_file=args.output_file)
    print("best params: ", hp.best_params)
    print("best result: ", hp.best_score)
    # print(hp.params2result[hp.params2str(hp.best_params)])

def ray_tune(args):
    config_file_path = f'./yaml_dir/{args.model_name}.yaml'
    config_file_list = (
        config_file_path.strip().split(" ") if config_file_path else None
    )
    config_file_list = (
        [os.path.join(os.getcwd(), file) for file in config_file_list]
        if config_file_path
        else None
    )
    
    param_file_path = f'./hyper/{args.model_name}.test'
    params_file = (
        os.path.join(os.getcwd(), param_file_path) if param_file_path else None
    )
    ray.init()
    tune.register_trainable("train_func", objective_function)
    config = {}
    with open(params_file, "r") as fp:
        for line in fp:
            para_list = line.strip().split(" ")
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = (
                para_list[0],
                para_list[1],
                "".join(para_list[2:]),
            )
            if para_type == "choice":
                para_value = eval(para_value)
                config[para_name] = tune.choice(para_value)
            elif para_type == "uniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.uniform(float(low), float(high))
            elif para_type == "quniform":
                low, high, q = para_value.strip().split(",")
                config[para_name] = tune.quniform(float(low), float(high), float(q))
            elif para_type == "loguniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.loguniform(
                    math.exp(float(low)), math.exp(float(high))
                )
            else:
                raise ValueError("Illegal param type [{}]".format(para_type))
    # choose different schedulers to use different tuning optimization algorithms
    # For details, please refer to Ray's official website https://docs.ray.io
    scheduler = ASHAScheduler(
        metric="recall@10", mode="max", max_t=10, grace_period=1, reduction_factor=2
    )

    local_dir = "./ray_log"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=5,
        log_to_file="ray_example.result",
        scheduler=scheduler,
        local_dir=local_dir,
        resources_per_trial={"gpu": 1},
    )

    best_trial = result.get_best_trial("recall@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MultiVAE", help="model name")
    parser.add_argument("--tool", type=str, default="Hyperopt", help="tuning tool")
    parser.add_argument("--output_name", type=str, default="output", help="output file name")
    parser.add_argument("--add_yaml", type=bool, default=False, help="add basic information to yaml file (only for the first time)")
    parser.add_argument("--algo", type=str, default="exhaustive", help="searching algorithm")
    args, _ = parser.parse_known_args()

    args.config_file = os.path.join("./yaml_dir", f'{args.model_name}.yaml')
    args.params_file = os.path.join("./hyper", f'{args.model_name}.test')
    args.output_file = os.path.join("./tuning_result", f'{args.model_name}_{args.output_name}.result')
    args.display_file = os.path.join("./tuning_result_display", f'{args.model_name}.html')

    if args.add_yaml == True:
        basic_info = f"""
data_path: dataset/
dataset: train_data
model: {args.model_name}
ongoing_file: "./tuning_result/{args.model_name}_ongoing.result"
"""
        with open(args.config_file, 'a') as f:
            f.write(basic_info)

    if args.tool == "Hyperopt":
        hyperopt_tune(args)
    elif args.tool == "Ray":
        ray_tune(args)
    else:
        raise ValueError(f"The tool [{args.tool}] should in ['Hyperopt', 'Ray']")

if __name__ == "__main__":
    main()