echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "*****************************************************"
echo "                SOCIALAI.STUDIO LCS2                 "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "======================================================"


export PYTHONPATH=":/data/data_store/rabiul/codes/multilingual-hate-speech"
DATASET=$1
SAMPLE_SZ=$2

if [ $DATASET == hasoc2020 ] 
then
    LANG_LIST=(en hi de)
elif [ $DATASET == hateval2019 ]
then
    LANG_LIST=es
elif [ $DATASET == evalita2020 ]
then
    LANG_LIST=it
else 
    LANG_LIST=(tr ar da gr)
fi

# process raw data files
# for LANG in ${LANG_LIST[@]}
# do
#     for SPLIT in train val test
#     do
#         python3 process_data.py --src_pkl ${DATASET}${LANG}_${SPLIT}.pkl --lang ${LANG} --force
#     done
# done

# process few-shot samples e.g. 128, 256, 512 
for LANG in ${LANG_LIST[@]}
do
    python3 pick_few_shots.py --src_pkl ${DATASET}${LANG}_train.pkl --dest_pkl ${DATASET}${LANG}_${SAMPLE_SZ}_train.pkl \
    --lang ${LANG} --shots $SAMPLE_SZ --rng_seed 42 --sampling maximize --force
    # python3 pick_few_shots.py --src_pkl ${DATASET}_val.pkl --dest_pkl ${DATASET}${LANG}_${SAMPLES}_val.pkl \
    # --lang ${LANG} --shots $SAMPLE_SZ --rng_seed 42 --sampling maximize --force
done
