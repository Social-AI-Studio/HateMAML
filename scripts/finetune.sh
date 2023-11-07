echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


# export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
export CUDA_VISIBLE_DEVICES=1
BATCH=32
LR=5e-5
# MODEL_TYPE=mbert
# MODEL_PATH=bert-base-multilingual-uncased
MODEL_TYPE=xlmr
MODEL_PATH=xlm-roberta-base
DATASET=founta
BASE=founta

if [ $DATASET == hasoc2020 ] 
then
    LANG_LIST=(hi de)
elif [ $DATASET == hateval2019 ]
then
    LANG_LIST=es
elif [ $DATASET == evalita2020 ]
then
    LANG_LIST=it
else 
    LANG_LIST=(tr ar da gr)
fi

FT_CHOICE=$1
META_SAMPLES=$2

# get the base model from english
finetune_english_only(){
    python3 src/maml/finetune.py --source_lang en --target_lang en --num_train_epochs 7 --meta_lr $LR \
    --seed 1 --model_name_or_path $MODEL_PATH --device_id 1 --batch_size $BATCH  --dataset_name $DATASET \
    --finetune_type $FT_CHOICE --overwrite_cache 
}

# use this script for finetuning
finetune_mono(){
    for T_LANG in ${LANG_LIST[@]}
    do
        for ID in 1 2 3 4 5
        do
            python3 src/maml/finetune.py --source_lang en --target_lang $T_LANG --num_train_epochs 5 --meta_lr $LR \
            --seed $ID --model_name_or_path $MODEL_PATH --device_id 1 --batch_size $BATCH  --dataset_name $DATASET \
            --finetune_type $FT_CHOICE --load_saved_base_model --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1 \
            --num_meta_samples $META_SAMPLES --overwrite_cache 
        done
    done
}

# finetune scale to all languages
CUR_LANGS=ar,da,gr,tr,hi,de,es,it
# CUR_LANGS=ar,da,hi,de,es,it
finetune_all(){
    EXP=finetune_collate 
    for ID in 1 2 3 4 5
    do
        python3 src/maml/hmaml_scale.py --num_train_epochs 7 --meta_lr $LR --model_name_or_path $MODEL_PATH --seed $ID \
        --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS --device_id 1 --overwrite_cache --num_meta_samples $META_SAMPLES  \
        --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1 
    done
}
EXCLUDE_LANGS=hi,gr,tr,ar
finetune_domain(){
    EXP=finetune_domain
    for ID in 1 2 3 4 5
    do
        python3 src/maml/hmaml_scale.py --num_train_epochs 5 --meta_lr $LR --model_name_or_path $MODEL_PATH --seed $ID \
        --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS --device_id 1 --overwrite_cache --num_meta_samples $META_SAMPLES \
        --base_model_path runs/finetune/${BASE}/en/en_ft/${MODEL_TYPE}/seed1 --exclude_langs $EXCLUDE_LANGS
    done
}

if [ $FT_CHOICE == "fft" ]
then
    finetune_mono 
elif [ $FT_CHOICE == "en_ft" ]
then
    finetune_english_only
elif [ $FT_CHOICE == "finetune_domain" ]
then
    finetune_domain
else
    finetune_all
fi
