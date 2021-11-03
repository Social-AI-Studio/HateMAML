echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
export CUDA_VISIBLE_DEVICES=0,1
BATCH=32
SHOTS=32
LR=2e-6
M_EPOCHS=4
MODEL_TYPE=Xlmr
MODEL_PATH=xlm-roberta-base

# use this script for HateMAML zeroshot refinement
# for T_LANG in ar tr da gr
# do
#     for ID in 1 2 3 4 5
#     do 
#         python3 src/maml/hmaml_vanilla_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#         --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/tanmoy/Mbert${ID}.ckpt \
#         --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-zero-refine --device_id 0 --overwrite_cache --refine_threshold 0.85
#     done
# done

# use this script for HateMAML zeroshot mixed refinement
# for T_LANG in ar tr da gr
# do
#     for ID in 1 2 3 4
#     do 
#         python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#         --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/tanmoy/Mbert${ID}.ckpt \
#         --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-zero-refine --device_id 0 --overwrite_cache --refine_threshold 0.85
#     done
# done

# train step 1+2 Hate_MAML using this script
# for T_LANG in tr ar gr da
# do
#     for A_LANG in ar tr gr da 
#     do
#         for TYPE in few 
#         do 
#             for ID in 1 2 3 4 5 
#             do 
#             if [ $A_LANG != $T_LANG ]
#             then
#                 python3 src/maml/hmaml_mixer_lit.py --source_lang en --aux_lang $A_LANG --target_lang $T_LANG --num_train_epochs 10 \
#                 --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/semeval2020/Mbert${ID}.ckpt \
#                 --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-zeroshot --device_id 0 --finetune_fewshot $TYPE --overwrite_cache
#             fi
#             done
#         done
#     done
# done


# work on maml
# for T_LANG in ar gr da tr
# do
#     for ID in 1 2 3 4 5 
#     do 
#         python3 src/maml/hmaml_vanilla_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#         --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/semeval2020/Mbert${ID}.ckpt \
#         --shots $SHOTS --batch_size $BATCH --exp_setting maml --device_id 0
#     done 
# done

# work on hmaml-fewshot
for T_LANG in ar gr da tr
do
    for ID in 1 2 3 4 5 
    do 
        python3 src/maml/hmaml_vanilla_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
        --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/semeval2020/${MODEL_TYPE}${ID}.ckpt \
        --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 0,1
    done 
done


# LANG=ar
# DATASET=semeval2020
# process raw data files
# python3 process_data.py --src_pkl ${DATASET}${LANG}val.pkl --lang ${LANG} --force

# process few-shot samples for MAML training
# python3 pick_few_shots.py --src_pkl ${DATASET}${LANG}_val.pkl --dest_pkl ${DATASET}${LANG}_few_val.pkl --lang ${LANG}  --shots 50 --rng_seed 42 --sampling equal --force

