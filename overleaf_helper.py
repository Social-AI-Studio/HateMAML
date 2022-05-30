import os
import json
import statistics
import numpy as np


def get_mean_stdv(data: list):
    mean = statistics.mean(data)
    stdv = statistics.stdev(data)
    # print(f"mean stdv {mean:.3f}_{{{stdv:.3f}}}")
    return mean, stdv


def read_files(dir_path: str):
    files = os.listdir(dir_path)
    bdict = {}
    identifier = "zeroshot"
    model_type = "bert"
    few_flag = True
    shot_flag = False
    meta_samples = 200
    shots = 10
    TYPE = "few"
    EPOCH = 5

    if "semeval" in dir_path:
        langs = ["ar", "da", "gr", "tr"]
    elif "hasoc" in dir_path:
        langs = ["hi", "de"]
    else:
        langs = ["es"]

    for fname in files:
        model_name = fname.split("_")[0]
        if model_type not in model_name:
            continue
        with open(os.path.join(dir_path, fname)) as f:
            data = json.load(f)

        if identifier == "zeroshot" and identifier in str(fname):
            aux_lang = fname.split("_")[-2]
            target_lang = fname.split("_")[-1].strip(".json")

            if few_flag and f"{shots}_{TYPE}" not in fname:
                continue

            if few_flag:
                epoch = int(fname.split("_")[-5])
            else:
                epoch = int(fname.split("_")[-4])

            if epoch == EPOCH:
                type = "hmaml-zeroshot"
                if few_flag:
                    type = TYPE

                print(f"Filtering on type = {type} epoch {epoch}, fname = {fname}")
                # print(json.dumps(data, indent=2))

                f1 = data[type].get("f1")

                key = aux_lang + "_" + target_lang
                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][key] = f1
        elif identifier == "zero-refine" and identifier in str(fname):
            target_lang = fname.split("_")[-1][:2]
            if "_" + str(shots) + "_" + target_lang not in str(fname):
                continue
            epoch = int(fname.split("_")[-3])
            if epoch == EPOCH:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["hmaml-zero-refine"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

        elif identifier == "_maml" and identifier in str(fname):
            target_lang = fname.split("_")[-1][:2]
            if "_" + str(shots) + "_" + target_lang not in str(fname):
                continue
            if shot_flag and str(meta_samples) not in str(fname):
                continue

            if shot_flag:
                epoch = int(fname.split("_")[-4])
            else:
                epoch = int(fname.split("_")[-3])

            if epoch == EPOCH:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["maml"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

        elif identifier == "fewshot" and identifier in str(fname):
            target_lang = fname.split("_")[-1][:2]
            if "_" + str(shots) + "_" + target_lang not in str(fname):
                continue
            if shot_flag and str(meta_samples) not in str(fname):
                continue

            if shot_flag:
                epoch = int(fname.split("_")[-4])
            else:
                epoch = int(fname.split("_")[-3])
            if epoch == EPOCH:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["hmaml-fewshot"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

    print(json.dumps(bdict, indent=2))
    print(dir_path)
    if identifier == "zeroshot":
        for tgt in langs:
            for aux in langs:
                if tgt == aux:
                    continue
                key = aux + "_" + tgt
                f1_list = []
                for bname in bdict.keys():
                    bdata = bdict[bname]
                    val = bdata.get(key)
                    if val:
                        f1_list.append(val)
                print(key, f1_list)
                if len(f1_list) > 1:
                    mean, stdv = get_mean_stdv(f1_list)
                    res_txt = f"{key}: {mean:.3f}_{{{stdv:.3f}}}\t"
                    print(res_txt)

    else:
        for key in langs:
            f1_list = []
            for bname in bdict.keys():
                bdata = bdict[bname]
                val = bdata.get(key)
                if val:
                    f1_list.append(val)
            print(key, f1_list)
            if len(f1_list) > 1:
                mean, stdv = get_mean_stdv(f1_list)
                res_txt = f"{key}: {mean:.3f}_{{{stdv:.3f}}}\t"
                print(res_txt)


def scaling_summary_gen(dir_path):
    bdict = {}
    model_type = "bert"
    EPOCH = 60
    meta_langs = ["es", "ar", "da", "gr", "tr", "hi", "de", "news", "tweets"]
    files = os.listdir(dir_path)
    for fname in files:
        pass
        model_name = fname.split("_")[0]
        if model_type not in model_name:
            continue
        with open(os.path.join(dir_path, fname)) as f:
            data = json.load(f)
        lang_sz = fname.split("_")[-1][0]
        if lang_sz != "8":
            continue

        identifier = fname.split("_")[1]
        if bdict.get(identifier) is None:
            bdict[identifier] = {}

        if identifier == "hmaml-scale":
            epoch = int(fname.split("_")[-2])
            if epoch != EPOCH:
                continue
            print(f"Filtering on epoch {epoch}, fname = {fname}")
        else:
            print(f"Filtering on fname = {fname}")

        for lang_id in meta_langs:
            if bdict[identifier].get(lang_id) is None:
                bdict[identifier][lang_id] = []
            cur_f1 = data[lang_id]["f1"]
            bdict[identifier][lang_id].append(cur_f1)
    for key in bdict:
        # print(f"{key} --> {bdict[key]}")
        print(f"Reporting {key}")
        res_txt = ""
        all_f1s = []
        for item in bdict[key]:
            all_f1s.extend(bdict[key][item])
            mean, stdv = get_mean_stdv(bdict[key][item])
            res_txt += f"{item}: {mean:.3f}_{{{stdv:.3f}}}\t"
        mean, stdv = get_mean_stdv(all_f1s)
        res_txt += f"avg: {mean:.3f}_{{{stdv:.3f}}}"
        print(res_txt)


def baseline_report():
    dirname = "runs/baselines/"
    subdirs = os.listdir(dirname)
    for subdir in subdirs:
        subdir_path = os.path.join(dirname, subdir)
        print("\n##################################")
        print(f"Processing {subdir_path}")
        subsubdirs = os.listdir(subdir_path)
        for subsubdir in subsubdirs:
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            for prefix in ["few", "full"]:
                fpath_sub = os.path.join(subsubdir_path, prefix)
                if not os.path.exists(fpath_sub):
                    continue

                fnames = os.listdir(fpath_sub)
                f1_collection = []
                for fn in fnames:
                    if "result" not in fn:
                        continue
                    fname = os.path.join(fpath_sub, fn)
                    with open(fname, "r") as f:
                        data = json.load(f)
                    cur_f1 = data[0]["test_macro_f1"]
                    f1_collection.append(cur_f1)
                print(f"\n{fpath_sub} mean f1 is {np.mean(f1_collection)}")


def main():
    dir_path = "runs/summary/semeval2020/hmaml_mixer_lit"
    read_files(dir_path)

    # dir_path = "runs/summary/analyze/hmaml_scale_lit"
    # scaling_summary_gen(dir_path)
    # baseline_report()


if __name__ == "__main__":
    main()
