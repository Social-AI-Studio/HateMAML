import os
import json
import statistics

def get_mean_stdv(data: list):
    mean = statistics.mean(data)
    stdv = statistics.stdev(data)
    print(f"mean {mean} stdv {stdv}")


def read_files():
    root_dir = "runs/summary"
    files = os.listdir(root_dir)
    bdict = {}
    for fname in files:
        bert_name = fname.split("_")[3]
        aux_lang = fname.split("_")[-2]
        target_lang = fname.split("_")[-1].strip(".json")
        print(f"Processing {fname}")

        with open(os.path.join(root_dir, fname)) as f:
            data = json.load(f)

        try:
            zero_f1 = data["hmaml-zeroshot"].get("f1")
        except Exception as e:
            pass
        
        key = target_lang + "_" + aux_lang
        if bdict.get(bert_name) is None:
            bdict[bert_name] = {}

        if bdict[bert_name].get(key) is None:
            bdict[bert_name][key] = zero_f1
        else:
            if bdict[bert_name][key] < zero_f1:
                bdict[bert_name][key] = zero_f1
    print(json.dumps(bdict, indent=2))

    langs = ["ar", "tr", "gr", "da"]
    for tgt in langs:
        for aux in langs:
            if tgt == aux:
                continue
            key = tgt + "_" + aux
            f1_list = []
            for bname in bdict.keys():
                bdata = bdict[bname]
                val = bdata.get(key)
                if val:
                    f1_list.append(val)
            print(key, f1_list)
            if len(f1_list) > 1:
                get_mean_stdv(f1_list)



if __name__ == '__main__':    
    read_files()
