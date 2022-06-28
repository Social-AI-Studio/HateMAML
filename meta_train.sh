echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
export CUDA_VISIBLE_DEVICES=0
BATCH=32
SHOTS=32
LR=2e-5
FAST_LR=1e-4
M_EPOCHS=50
MODEL_TYPE=mbert
MODEL_PATH=bert-base-multilingual-uncased
DATASET=semeval2020
LIMIT=0.85
BASE=hasoc2020

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

# use this script for HateMAML zeroshot refinement
meta_train_refine(){
    for T_LANG in ${LANG_LIST[@]}
    do
        for ID in 1 2 3 4 5
        do 
            python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 7 \
            --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $FAST_LR --load_saved_base_model \
            --seed $ID --shots $SHOTS --batch_size $BATCH --exp_setting hmaml --metatune_type fewshot \
            --device_id 0 --overwrite_cache --refine_threshold 0.85 \
            # --num_meta_samples 500 --base_model_path runs/baselines/${BASE}/en/full/${MODEL_TYPE}1.ckpt 
        done
    done
}

# use this script for HateMAML zeroshot mixed refinement
meta_train_mixed_refine(){
    for T_LANG in ar da gr tr
    do
        for ID in 1 2 3 4 5
        do
            for EPOCHS in 10
            do 
                python3 src/maml/hmaml_vanilla_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
                --num_meta_iterations $EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model \
                --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt --model_name_or_path $MODEL_PATH --shots $SHOTS \
                --batch_size $BATCH --exp_setting hmaml-zero-refine --device_id 0 --refine_threshold 0.85 --dataset_name $DATASET
            done
        done
    done
}


# train step 1+2 Hate_MAML using this script
meta_train_mixed(){
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
                        --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $LR --load_saved_base_model \
                        --base_model_path runs/baselines/${BASE}/en/full/${MODEL_TYPE}1.ckpt --seed $ID --model_name_or_path $MODEL_PATH \
                        --shots $SHOTS --batch_size $BATCH --exp_setting hmaml --device_id 1 --dataset_name $DATASET \
                        --num_meta_samples 500 --metatune_type $TYPE --overwrite_cache 
                    fi
                    done
                done
            done
        done
    done
}


# work on maml
meta_auxiliary_refine(){
    for T_LANG in de
    do
        if [ $T_LANG == de ]
        then
            LIMIT=0.850
        elif [ $T_LANG == hi ]
        then
            LIMIT=0.825
        fi
        for SHOTS in 36 40
        do
            for ID in 1 2 3 4 5
            do 
                python3 src/maml/hmaml_mixer_lit.py --source_lang en --target_lang $T_LANG --num_train_epochs 6 \
                --num_meta_iterations $M_EPOCHS --meta_lr 4e-6 --fast_lr $LR --load_saved_base_model --base_model_path runs/${DATASET}/${MODEL_TYPE}${ID}.ckpt \
                --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET --refine_threshold $LIMIT --num_meta_samples 200 --overwrite_cache
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
            --batch_size $BATCH --exp_setting hmaml-fewshot --device_id 1 --dataset_name $DATASET
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


meta_train_all(){
    CUR_LANGS=ar,da,gr,tr,hi,de,es,it
    LANG_LIST=(ar da gr tr hi de es it)
    for LG in ${LANG_LIST[@]}
    do
        for ID in 1 2 3 4 5
        do
            python3 src/maml/hmaml_scale_lit.py --num_train_epochs 8 --num_meta_iterations $M_EPOCHS --meta_lr $LR --fast_lr $FAST_LR \
            --model_name_or_path $MODEL_PATH --shots $SHOTS --batch_size $BATCH --exp_setting $EXP --meta_langs $CUR_LANGS \
            --seed $ID --num_meta_samples $SAMPLE_SZ --device_id 1 --overwrite_cache
        done
    done
}

EXP=$1
SAMPLE_SZ=$2

meta_train_all
