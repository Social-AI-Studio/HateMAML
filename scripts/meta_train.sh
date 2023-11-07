echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


# export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
export CUDA_VISIBLE_DEVICES=1
BATCH=32
SHOTS=32
LR=5e-5
FAST_LR=3e-5
# MODEL_TYPE=mbert
# MODEL_PATH=bert-base-multilingual-uncased
MODEL_TYPE=xlmr
MODEL_PATH=xlm-roberta-base
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

# hmaml-fewshot = hmaml_{target} // vanilla
# hmaml-fewshot = hmaml_{src+target} // mixer
# hmaml-zero-refine = hmaml_{self_training} // vanilla
# hmaml-zero-refine = hmaml_{src+self_training} // mixer
# maml= MAML // vanilla


# use this script for HateMAML zeroshot mixed refinement
meta_refine(){
    EXP=hmaml
    TYPE=refine
    for T_LANG in ${LANG_LIST[@]}
    do
        for ID in 1 2 # 3 4 5 
        do
            python3 src/maml/hmaml_mixer.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 --num_meta_iterations $M_EPOCHS \
            --meta_lr $LR --fast_lr $FAST_LR --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1 \
            --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --device_id 1 --seed $ID \
            --dataset_name $DATASET --refine_threshold $LIMIT --num_meta_samples $SAMPLE_SZ --metatune_type $TYPE --overwrite_cache
        done
    done
}



# train step 1+2 Hate_MAML using this script
meta_train_mixed(){
    if [ $EXP == "xmaml" ]
    then
        script=src/maml/xmaml.py
    else
        script=src/maml/hmaml_mixer.py
    fi

    for T_LANG in ${LANG_LIST[@]}
    do
        for A_LANG in ${LANG_LIST[@]}
        do
            for TYPE in zeroshot
            do
                for ID in 1 2 # 3 4 5 
                do
                if [ $A_LANG != $T_LANG ]
                then
                    python3 $script --source_lang en --aux_lang $A_LANG --target_lang $T_LANG --num_train_epochs 5 \
                    --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $FAST_LR --seed $ID --model_name_or_path $MODEL_PATH \
                    --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1 --shots $SHOTS --batch_size $BATCH \
                    --exp_setting $EXP --device_id 1 --dataset_name $DATASET --num_meta_samples $SAMPLE_SZ --metatune_type $TYPE \
                    --overwrite_cache

                fi
                done
            done
        done
    done
}



# work on hmaml-fewshot
meta_train_fewshot(){
    for T_LANG in ${LANG_LIST[@]}
    do
        for ID in 1 2 3 4 5
        do
            python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
            --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model \
            --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt --model_name_or_path $MODEL_PATH --shots $SHOTS \
            --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET --seed $ID
        done
    done
}

# one model to support all languages
meat_train_progressive(){
    EXP=hmaml-scale # scale to all languages
    META_LANGS=( "ar" "da" "gr" "tr" "hi" "de" "es" )
    CUR_LANGS=""
    for L_ID in "${!META_LANGS[@]}"
    do
        if [ -z "$CUR_LANGS" ]
        then
            CUR_LANGS+="${META_LANGS[$L_ID]}"
        else
            CUR_LANGS+=",${META_LANGS[$L_ID]}"
        fi
        LANG_CNT=$(( L_ID + 1 ))
        M_EPOCHS=$(( LANG_CNT * 10 ))
        M_EPOCHS=$(( M_EPOCHS + 10 ))
        echo "Langs in use $CUR_LANGS --> epochs $M_EPOCHS"
        for ID in 1 2 3 4 5
        do
            python3 src/maml/hmaml_scale_lit.py --num_train_epochs 6 --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR \
                --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt --model_name_or_path $MODEL_PATH \
                --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS --device_id 1  --overwrite_cache
        done
    done
}


# CUR_LANGS=ar,da,hi,de,es,it
CUR_LANGS=ar,da,gr,tr,hi,de,es,it
EXCLUDE_LANGS=hi,gr,tr,ar
meta_train_all(){
    for ID in 1 2 3 4 5
    do
        python3 src/maml/hmaml_scale.py --num_train_epochs 8 --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $FAST_LR \
        --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS \
        --seed $ID --num_meta_samples $SAMPLE_SZ --device_id 1 --overwrite_cache --wandb_proj hatemaml \
        --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1
    done
}
meta_train_domain(){
    for ID in 1 2 3 4 5
    do
        python3 src/maml/hmaml_scale.py --num_train_epochs 8 --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $FAST_LR \
        --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS \
        --seed $ID --num_meta_samples $SAMPLE_SZ --device_id 1 --overwrite_cache --wandb_proj hatemaml \
        --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1  --exclude_langs $EXCLUDE_LANGS
    done
}
EXP=$1
SAMPLE_SZ=$2
M_EPOCHS=$3

if [ $EXP == "hmaml_scale" ]
then
    meta_train_all
elif [ $EXP == "hmaml_domain" ]
then
    meta_train_domain
elif [ $EXP == "hmaml_refine" ]
then
    meta_refine
else
    meta_train_mixed
fi
