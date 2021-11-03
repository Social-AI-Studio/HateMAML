import os
from random import shuffle


def variations_combo_gen(variations):
    if variations == {}:
        yield {}
    else:
        variations = dict(variations)
        target_key = sorted([(len(v), k) for k, v in variations.items()])[-1][1]
        target_vals = variations[target_key]
        del variations[target_key]
        for target_val in target_vals:
            for ret_dict in variations_combo_gen(variations):
                ret_dict[target_key] = target_val
                yield ret_dict


VARIATIONS = {
#    "epoch": [15],
    #"lr": [6e-7,2e-6,4e-6,8e-6],
    "lr": [2e-6],
#    "train_batch_size": [64],
    "rng_seed": [1,2,3,4,5,],
    "model_type":["mbert",],
    #"model_type,train_ckpt,rng_seed":[
    ##("xlmr", "fountaen/best/2021_10_23_02_31_47/epoch=2-val_macro_f1=0.934.ckpt", 1),
    ##("xlmr", "fountaen/best/2021_10_23_05_09_56/epoch=3-val_macro_f1=0.933.ckpt", 2),
    ##("xlmr", "fountaen/best/2021_10_23_07_20_54/epoch=1-val_macro_f1=0.934.ckpt", 3),
    ##("xlmr", "fountaen/best/2021_10_23_08_58_51/epoch=2-val_macro_f1=0.933.ckpt", 4),
    ##("xlmr", "fountaen/best/2021_10_23_10_50_42/epoch=4-val_macro_f1=0.934.ckpt", 5),
    #("mbert", "fountaen/best/2021_10_23_02_32_58/epoch=2-val_macro_f1=0.934.ckpt", 1),
    #("mbert", "fountaen/best/2021_10_23_04_45_08/epoch=2-val_macro_f1=0.932.ckpt", 2),
    #("mbert", "fountaen/best/2021_10_23_06_00_58/epoch=2-val_macro_f1=0.932.ckpt", 3),
    #("mbert", "fountaen/best/2021_10_23_07_28_42/epoch=2-val_macro_f1=0.935.ckpt", 4),
    #("mbert", "fountaen/best/2021_10_23_09_13_28/epoch=2-val_macro_f1=0.934.ckpt", 5),
    #],
}

all_variations = [i for i in variations_combo_gen(VARIATIONS)]
#shuffle(all_variations)

for variations_combo in all_variations:
    args_str = ""
    for arg in variations_combo.keys():
        if "," in arg:
            cs_args = arg.split(",")
            for cs_arg_num,cs_arg in enumerate(cs_args):
                args_str += f" --{cs_arg} {variations_combo[arg][cs_arg_num]}"    #assuming variations_combo[arg] is a tuple.
        else:
            args_str += f" --{arg} {variations_combo[arg]}"
    built_cmd = f"python3 baselines/simple-baselines.py --epochs 6 --max_seq_len 64 --es_patience 4 --dataset_name semeval2020da --train --batch_size 32 {args_str}"
    print(f"DRIVER (executing)>>{built_cmd}<<")
    ret_status = os.system(built_cmd)
    if ret_status != 0:
        print(f"DRIVER (non-zero exit status from execution)>>{ret_status}<<")
        exit()

for variations_combo in all_variations:
    args_str = ""
    for arg in variations_combo.keys():
        if "," in arg:
            cs_args = arg.split(",")
            for cs_arg_num,cs_arg in enumerate(cs_args):
                args_str += f" --{cs_arg} {variations_combo[arg][cs_arg_num]}"    #assuming variations_combo[arg] is a tuple.
        else:
            args_str += f" --{arg} {variations_combo[arg]}"
    built_cmd = f"python3 baselines/simple-baselines.py --epochs 6 --max_seq_len 64 --es_patience 4 --dataset_name semeval2020ar --train --batch_size 32 {args_str}"
    print(f"DRIVER (executing)>>{built_cmd}<<")
    ret_status = os.system(built_cmd)
    if ret_status != 0:
        print(f"DRIVER (non-zero exit status from execution)>>{ret_status}<<")
        exit()

for variations_combo in all_variations:
    args_str = ""
    for arg in variations_combo.keys():
        if "," in arg:
            cs_args = arg.split(",")
            for cs_arg_num,cs_arg in enumerate(cs_args):
                args_str += f" --{cs_arg} {variations_combo[arg][cs_arg_num]}"    #assuming variations_combo[arg] is a tuple.
        else:
            args_str += f" --{arg} {variations_combo[arg]}"
    built_cmd = f"python3 baselines/simple-baselines.py --epochs 6 --max_seq_len 64 --es_patience 4 --dataset_name semeval2020tr --train --batch_size 32 {args_str}"
    print(f"DRIVER (executing)>>{built_cmd}<<")
    ret_status = os.system(built_cmd)
    if ret_status != 0:
        print(f"DRIVER (non-zero exit status from execution)>>{ret_status}<<")
        exit()

for variations_combo in all_variations:
    args_str = ""
    for arg in variations_combo.keys():
        if "," in arg:
            cs_args = arg.split(",")
            for cs_arg_num,cs_arg in enumerate(cs_args):
                args_str += f" --{cs_arg} {variations_combo[arg][cs_arg_num]}"    #assuming variations_combo[arg] is a tuple.
        else:
            args_str += f" --{arg} {variations_combo[arg]}"
    built_cmd = f"python3 baselines/simple-baselines.py --epochs 6 --max_seq_len 64 --es_patience 4 --dataset_name semeval2020gr --train --batch_size 32 {args_str}"
    print(f"DRIVER (executing)>>{built_cmd}<<")
    ret_status = os.system(built_cmd)
    if ret_status != 0:
        print(f"DRIVER (non-zero exit status from execution)>>{ret_status}<<")
        exit()
