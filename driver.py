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
    "epoch": [7],
    "lr": [2e-5],
    # "train_batch_size": [32],
    "rng_seed": [1, 2, 3, 4, 5],
    "model_type": [
        "xlmr",
    ],
    # "model_type,train_ckpt,rng_seed,freeze_layers":[
    # ("xlmr", "fountaen/best/2021_11_06_15_26_15/epoch=4-val_macro_f1=0.933.ckpt", 1),
    # ("xlmr", "fountaen/best/2021_11_06_16_05_08/epoch=4-val_macro_f1=0.933.ckpt", 2),
    # ("xlmr", "fountaen/best/2021_11_06_16_44_07/epoch=4-val_macro_f1=0.935.ckpt", 3),
    # ("xlmr", "fountaen/best/2021_11_06_17_23_09/epoch=4-val_macro_f1=0.932.ckpt", 4),
    # ("xlmr", "fountaen/best/2021_11_06_18_02_11/epoch=5-val_macro_f1=0.934.ckpt", 5),
    # ("mbert", "fountaen/best/2021_10_23_02_32_58/epoch=2-val_macro_f1=0.934.ckpt", 1, "top6"),
    # ("mbert", "fountaen/best/2021_10_23_04_45_08/epoch=2-val_macro_f1=0.932.ckpt", 2, "top6"),
    # ("mbert", "fountaen/best/2021_10_23_06_00_58/epoch=2-val_macro_f1=0.932.ckpt", 3, "top6"),
    # ("mbert", "fountaen/best/2021_10_23_07_28_42/epoch=2-val_macro_f1=0.935.ckpt", 4, "top6"),
    # ("mbert", "fountaen/best/2021_10_23_09_13_28/epoch=2-val_macro_f1=0.934.ckpt", 5, "top6"),
    # ],
}

all_variations = [i for i in variations_combo_gen(VARIATIONS)]

datasets_name_collection = [
    "founta",
    "semeval2020",
    # "hateval2019",
    # "hasoc2020",
    # "evalita2020",
]
lang_list_map = {
    "semeval2020": ["ar", "da", "tr", "gr"],
    "hateval2019": ["es"],
    "hasoc2020": ["en", "hi", "de"],
    "founta": ["en"],
    "evalita2020": ["en"],
}

for fewshot_flag in [False]:
    for dataset_name in datasets_name_collection:
        lg_list = lang_list_map[dataset_name]
        for lang in lg_list:
            for variations_combo in all_variations:
                args_str = ""
                for arg in variations_combo.keys():
                    if "," in arg:
                        cs_args = arg.split(",")
                        for cs_arg_num, cs_arg in enumerate(cs_args):
                            args_str += f" --{cs_arg} {variations_combo[arg][cs_arg_num]}"  # assuming variations_combo[arg] is a tuple.
                    else:
                        args_str += f" --{arg} {variations_combo[arg]}"
                if fewshot_flag and lang != "en":
                    built_cmd = f"python3 baselines/simple-baselines.py --max_seq_len 128 --es_patience 3 --dataset_name {dataset_name} --lang {lang} --train --batch_size 32 --num_workers 24 --fewshot --test_splits val,test {args_str}"
                else:
                    built_cmd = f"python3 baselines/simple-baselines.py --max_seq_len 128 --es_patience 3 --dataset_name {dataset_name} --lang {lang} --train --batch_size 32 --num_workers 24 --test_splits val,test {args_str}"

                print(f"DRIVER (executing)>>{built_cmd}<<")
                ret_status = os.system(built_cmd)
                if ret_status != 0:
                    print(f"DRIVER (non-zero exit status from execution)>>{ret_status}<<")
                    exit()
