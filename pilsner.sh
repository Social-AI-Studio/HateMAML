echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
export CUDA_VISIBLE_DEVICES=1
BATCH=32
SHOTS=16
LR=3e-5
M_EPOCHS=30
MODEL_TYPE=mbert
MODEL_PATH=bert-base-multilingual-uncased
DATASET=semeval2020
LIMIT=0.85
BASE=founta

if [ $DATASET == hasoc2020 ] 
then
    LANG_LIST=(hi de)
elif [ $DATASET == hateval2019 ]
then
    LANG_LIST=es
else 
    LANG_LIST=(tr ar da gr)
fi

# use this script for HateMAML zeroshot refinement
# for T_LANG in ${LANG_LIST[@]}
# do
#     for ID in 1 2 3 4 5
#     do 
#         python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 7 --num_meta_samples 500 \
#         --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $LR --load_saved_base_model \
#         --base_model_path runs/baselines/${BASE}/en/full/${MODEL_TYPE}1.ckpt --seed $ID --shots $SHOTS --batch_size $BATCH \
#         --exp_setting xmetra --metatune_type fewshot --device_id 0 --overwrite_cache --refine_threshold 0.85
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

# use this script for HateMAML zeroshot mixed refinement
# for T_LANG in ${LANG_LIST[@]}
# do
#     for ID in 1 2 3 4 5
#     do
#         python3 src/maml/finetune.py --source_lang en --target_lang $T_LANG --num_train_epochs 7 --num_meta_samples 500 \
#         --meta_lr $LR --load_saved_base_model --base_model_path runs/baselines/${BASE}/en/full/${MODEL_TYPE}1.ckpt --seed $ID \
#         --model_name_or_path $MODEL_PATH --device_id 1 --batch_size $BATCH  --dataset_name $DATASET --finetune_fewshot few --overwrite_cache
#     done
# done

# train step 1+2 Hate_MAML using this script
for T_LANG in ${LANG_LIST[@]}
do
    for A_LANG in ${LANG_LIST[@]}
    do
        for TYPE in zeroshot
        do
            for MODEL_PATH in bert-base-multilingual-uncased # xlm-roberta-base
            do
            if [ $MODEL_PATH == xlm-roberta-base ]
            then
                MODEL_TYPE=xlmr
            else
                MODEL_TYPE=mbert
            fi 
                for ID in 1 2 3 4 5 
                do 
                if [ $A_LANG != $T_LANG ]
                then
                    python3 src/maml/hmaml_mixer_lit.py --source_lang en --aux_lang $A_LANG --target_lang $T_LANG --num_train_epochs 10 \
                    --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $LR --load_saved_base_model --base_model_path runs/baselines/${BASE}/en/full/${MODEL_TYPE}1.ckpt --seed $ID \
                    --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml --device_id 1 --dataset_name $DATASET \
                    --num_meta_samples 500 --overwrite_cache --metatune_type $TYPE
                fi
                done
            done
        done
    done
done


# work on maml
# for T_LANG in de
# do
#     if [ $T_LANG == de ]
#     then
#         LIMIT=0.850
#     elif [ $T_LANG == hi ]
#     then
#         LIMIT=0.825
#     fi
#     for SHOTS in 36 40
#     do
#         for ID in 1 2 3 4 5
#         do 
#             python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#             --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
#             --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET --refine_threshold $LIMIT --num_meta_samples 200 --overwrite_cache
#         done 
#     done
# done

# hmaml-fewshot = hmaml_{target} // vanilla
# hmaml-fewshot = hmaml_{src+target} // mixer
# hmaml-zero-refine = hmaml_{self_training} // vanilla
# hmaml-zero-refine = hmaml_{src+self_training} // mixer
# maml= MAML // vanilla


# work on hmaml-fewshot
# for T_LANG in ${LANG_LIST[@]}
# do
#     for ID in 1 2 3 4 5
#     do 
#         python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
#         --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
#         --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET
#     done 
# done

# LANG=it
# DATASET=semeval2020
# SAMPLES=200
# process raw data files
# for SPLIT in test # train val
# do
    # python3 process_data.py --src_pkl ${DATASET}${LANG}_${SPLIT}.pkl --lang ${LANG} --force
    # python3 process_data.py --src_pkl ${DATASET}tweets_${SPLIT}.pkl --lang ${LANG} --force
# done

# process few-shot samples for MAML training
# python3 pick_few_shots.py --src_pkl ${DATASET}${LANG}_train.pkl --dest_pkl ${DATASET}${LANG}_${SAMPLES}_train.pkl --lang ${LANG}  --shots $SAMPLES --rng_seed 42 --sampling maximize --force
# python3 pick_few_shots.py --src_pkl ${DATASET}_val.pkl --dest_pkl ${DATASET}${LANG}_${SAMPLES}_val.pkl --lang ${LANG}  --shots $SAMPLES --rng_seed 42 --sampling maximize --force

# one model to support all languages 
# EXP=finetune-collate # scale to all languages
# EXP=hmaml-scale # scale to all languages
# META_LANGS=( "ar" "da" "gr" "tr" "hi" "de" "es" )
# CUR_LANGS=""
# for L_ID in "${!META_LANGS[@]}" 
# do 
#     if [ -z "$CUR_LANGS" ]
#     then
#         CUR_LANGS+="${META_LANGS[$L_ID]}"
#     else
#         CUR_LANGS+=",${META_LANGS[$L_ID]}"
#     fi
#     LANG_CNT=$(( L_ID + 1 ))
#     M_EPOCHS=$(( LANG_CNT * 10 ))
#     M_EPOCHS=$(( M_EPOCHS + 10 ))
#     echo "Langs in use $CUR_LANGS --> epochs $M_EPOCHS"
#     for ID in 1 2 3 4 5
#     do
#         python3 src/maml/hmaml_scale_lit.py --num_train_epochs 6 --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR \
#             --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt --model_name_or_path $MODEL_PATH \
#             --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS --device_id 1  --overwrite_cache
#     done
# done
# CUR_LANGS=ar,da,gr,tr,hi,de,es,it
# LANG_LIST=(ar da gr tr hi de es it)
# for LG in ${LANG_LIST[@]}
# do
#     for ID in 1 2 3 4 5
#     do
#         python3 src/maml/hmaml_scale_lit.py --num_train_epochs 8 --num_meta_iterations $M_EPOCHS --meta_lr 4e-5 --fast_lr $LR \
#             --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt --model_name_or_path $MODEL_PATH \
#             --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --meta_langs $LG --device_id 1  --overwrite_cache
#     done
# done
