import os
from random import shuffle

def variations_combo_gen(variations):
    if variations == {}: yield {}
    else:
        variations = dict(variations)
        target_key = sorted([(len(v),k) for k,v in variations.items()])[-1][1]
        target_vals = variations[target_key]
        del variations[target_key]
        for target_val in target_vals:
            for ret_dict in variations_combo_gen(variations):
                    ret_dict[target_key] = target_val
                    yield ret_dict

VARIATIONS = {
    'epoch':[15],
    'lr':[2e-3,2e-4,2e-5,2e-6],
    #'gama':[0.0,0.25,0.5,0.75,1.0,],
    #'gama':[0.0,0.25,0.5,],
    'gama':[0.75,1.0,],
    'train_batch_size':[64],
    'embedding_dim':[300],
    'hidden_size':[64],
}

all_variations = [i for i in variations_combo_gen(VARIATIONS)]
shuffle(all_variations)

for variations_combo in all_variations:
    args_str = ''
    for arg in variations_combo.keys():
        args_str += f" --{arg} {variations_combo[arg]}"
    #built_cmd = f"python3 main.py --train_data_path dataset/offensevalolid2019_nocpt_train.tsv --dev_data_path dataset/offensevalolid2019_nocpt_val.tsv --test_data_path dataset/offensevalolid2019_nocpt_test.tsv --txt_embedding_path dataset/glove.6B.300d.txt --cpt_embedding_path dataset/glove.6B.300d.txt {args_str}"
    built_cmd = f"python3 main.py --train_data_path dataset/hateval2019en_train.tsv --dev_data_path dataset/hateval2019en_val.tsv --test_data_path dataset/hateval2019en_test.tsv --txt_embedding_path dataset/numberbatch/numberbatch-en-19.08.txt --cpt_embedding_path dataset/numberbatch/numberbatch-en-19.08.txt {args_str}"
    print(f"DRIVER (executing)>>{built_cmd}<<")
    ret_status = os.system(built_cmd)
    if ret_status != 0:
        print(f"DRIVER (non-zero exit status from execution)>>{ret_status}<<")
        exit()
