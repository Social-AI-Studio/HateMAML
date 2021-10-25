export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"

# train MAML using this script
#python3 src/maml/maml_two_step.py --base_lang en --meta_train_lang es --num_train_epochs 3 --num_meta_iterations 10 --learning_rate 2e-6 

# train step 1 HateMAML (domain generalization) using this script
# python3 src/maml/maml_two_step.py --source_lang en --aux_lang tr --target_lang da \
# --num_train_epochs 3 --num_meta_iterations 5 --meta_lr 4e-6 --fast_lr 2e-6 \
# --load_saved_base_model --batch_size 32 --exp_setting hmaml-step1

# use this script for HateMAML zeroshot refinement
python3 src/maml/maml_two_step.py --dataset_name hateval2019 --source_lang en --aux_lang es  --target_lang es \
--num_train_epochs 3 --num_meta_iterations 5 --meta_lr 4e-6 --fast_lr 4e-6 \
--overwrite_base_model --batch_size 32 --exp_setting hmaml-zero-refine


# train step 1+2 Hate_MAML using this script
#python3 src/maml/maml_two_step.py --base_lang en --meta_train_lang es --num_train_epochs 3 --num_meta_iterations 10 --learning_rate 2e-6 


# LANG=es
# DATASET=hateval2019
# process raw data files
# python3 process_data.py --src_pkl ${DATASET}${LANG}_test.pkl --lang ${LANG} --force

# process few-shot samples for MAML training
# python3 pick_few_shots.py --src_pkl ${DATASET}${LANG}_train.pkl --dest_pkl ${DATASET}${LANG}_few_train.pkl --lang ${LANG}  --shots 500 --rng_seed 42 --sampling stratified --force

