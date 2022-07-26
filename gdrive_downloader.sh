#!/bin/bash
# fileid="1nrRZ7ON_cEs7fynihXq65sPdhWxxvwKz"
# filename="runs/semeval2020/Mbert5.ckpt"
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}


#zip download
fileid="16tvOVzvHwfv5Z_poHPV7Ts7nnYSzUlBJ"
filename="runs/hateval2019/mbert.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}