echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
export CUDA_VISIBLE_DEVICES=1
BATCH=32
SHOTS=10
LR=2e-5
M_EPOCHS=15
MODEL_TYPE=Mbert
MODEL_PATH=bert-base-multilingual-uncased
DATASET=hasoc2020
LIMIT=0.85

if [ $DATASET == hasoc2020 ] 
then
    LANG_LIST=(hi de)
elif [ $DATASET == hateval2019 ]
then
    LANG_LIST=es
else 
    LANG_LIST=(ar da gr tr)
fi

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
# for T_LANG in ar da gr tr
# do
#     for ID in 1 2 3 4 5
#     do
#         for EPOCHS in 10
#         do 
#             python3 src/maml/hmaml_vanilla_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#             --num_meta_iterations $EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
#             --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-zero-refine --device_id 0 --refine_threshold 0.85 --dataset_name $DATASET
#         done
#     done
# done

# train step 1+2 Hate_MAML using this script
# for T_LANG in ${LANG_LIST[@]}
# do
#     for A_LANG in ${LANG_LIST[@]}
#     do
#         for TYPE in few full
#         do
#             for MODEL_PATH in bert-base-multilingual-uncased xlm-roberta-base
#             do
#             if [ $MODEL_PATH == xlm-roberta-base ]
#             then
#                 MODEL_TYPE=Xlmr
#             else
#                 MODEL_TYPE=Mbert
#             fi 
#                 for ID in 1 2 3 4 5 
#                 do 
#                 if [ $A_LANG != $T_LANG ]
#                 then
#                     python3 src/maml/hmaml_mixer_lit.py --source_lang en --aux_lang $A_LANG --target_lang $T_LANG --num_train_epochs 10 \
#                     --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
#                     --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-zeroshot --device_id 1 --dataset_name $DATASET \
#                     --num_meta_samples 200 --overwrite_cache --finetune_fewshot $TYPE
#                 fi
#                 done
#             done
#         done
#     done
# done


# work on maml
for T_LANG in de
do
    if [ $T_LANG == de ]
    then
        LIMIT=0.850
    elif [ $T_LANG == hi ]
    then
        LIMIT=0.825
    fi
    for SHOTS in 4 8 12 16 20 24 28 32
    do
        for ID in 1 2 3 4 5
        do 
            python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
            --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
            --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET --refine_threshold $LIMIT --num_meta_samples 200 --overwrite_cache
        done 
    done
done

# hmaml-fewshot = hmaml_{target} // vanilla
# hmaml-fewshot = hmaml_{src+target} // mixer
# hmaml-zero-refine = hmaml_{self_training} // vanilla
# hmaml-zero-refine = hmaml_{src+self_training} // mixer
# maml= MAML // vanilla


# work on hmaml-fewshot
# for T_LANG in de hi
# do
#     for ID in 1 2 3 4 5
#     do 
#         python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#         --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
#         --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET
#     done 
# done


# LANG=da
# DATASET=semeval2020
# SAMPLES=500
# process raw data files
# for SPLIT in train test val
# do
#     python3 process_data.py --src_pkl ${DATASET}${LANG}_${SPLIT}.pkl --lang ${LANG} --force
# done

# process few-shot samples for MAML training
# python3 pick_few_shots.py --src_pkl ${DATASET}${LANG}_train.pkl --dest_pkl ${DATASET}${LANG}_${SHOTS}_train.pkl --lang ${LANG}  --shots $SAMPLES --rng_seed 42 --sampling maximize --force

