
import os
import json
import statistics
import numpy as np
import argparse


def get_mean_stdv(data: list):
    mean = statistics.mean(data)
    stdv = statistics.stdev(data)
    # print(f"mean stdv {mean:.3f}_{{{stdv:.3f}}}")
    return mean, stdv


def meta_tuning_summary(args, dir_path: str):
    files = os.listdir(dir_path)
    bdict = {}

    if "semeval" in dir_path:
        langs = ["ar", "da", "gr", "tr"]
    elif "hasoc" in dir_path:
        langs = ["hi", "de"]
    else:
        langs = ["es"]

    for fname in files:
        model_name = fname.split("_")[0]
        if args.exp_setting not in fname:
            continue
        if args.model_type not in model_name:
            continue
        with open(os.path.join(dir_path, fname)) as f:
            data = json.load(f)

        if args.type == "metatune":
            aux_lang = fname.split("_")[-2]
            target_lang = fname.split("_")[-1].strip(".json")
            shots = fname.split("_")[2]
            if "zeroshot_" + aux_lang not in fname:
                continue
            if shots == args.shots:
                print(f"Filtering on type = {args.type}, fname = {fname}")
                # print(json.dumps(data, indent=2))
                result = data["result"]
                f1 = -1
                for info in result:
                    cur_f1 = info["meta"]["f1"]
                    if f1 < cur_f1:
                        f1 = cur_f1

                key = aux_lang + "_" + target_lang
                if bdict.get(model_name) is None:
                    bdict[model_name] = {}
                bdict[model_name][key] = f1
        elif args.type == "refine":
            target_lang = fname.split("_")[-1][:2]
            if "_" + str(args.shots) + "_" + args.type not in str(fname):
                continue
            print(f"Filtering on fname = {fname}")
            # print(json.dumps(data, indent=2))
            result = data["result"]
            f1 = -1
            for info in result:
                cur_f1 = info["meta"]["f1"]
                if f1 < cur_f1:
                    f1 = cur_f1
            if bdict.get(model_name) is None:
                bdict[model_name] = {}

            bdict[model_name][target_lang] = f1
        elif args.type == "zeroshot":
            target_lang = fname.split("_")[-1][:2]
            if "_" + str(args.shots)  not in str(fname) and target_lang not in str(fname):
                continue
            print(f"Filtering on fname = {fname}")
            # print(json.dumps(data, indent=2))
            f1 = data["zero"]["f1"]
            if bdict.get(model_name) is None:
                bdict[model_name] = {}

            bdict[model_name][target_lang] = f1

    print(json.dumps(bdict, indent=2))
    print(dir_path)
    if args.type == "metatune":
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
    meta_langs = ["es", "ar", "da", "gr", "tr", "hi", "de", "news", "tweets"]
    for mname in os.listdir(dir_path):
        model_dir = os.path.join(dir_path, mname)
        for seedname in os.listdir(model_dir):
            seed_dir = os.path.join(model_dir, seedname)
            for expname in os.listdir(seed_dir):
                exp_dir = os.path.join(seed_dir, expname)
                for szname in os.listdir(exp_dir):
                    sz_dir = os.path.join(exp_dir, szname)
                    for fname in os.listdir(sz_dir):
                        with open(os.path.join(sz_dir, fname)) as f:
                            data = json.load(f)

                        if bdict.get(expname) is None:
                            bdict[expname] = {}
                        if bdict[expname].get(mname) is None:
                            bdict[expname][mname] = {}
                        if bdict[expname][mname].get(szname) is None:
                            bdict[expname][mname][szname] = {}


                        curr = bdict[expname][mname][szname]
                        best_avg = 0
                        best_info = None
                        iterr = None
                        if "result" in data and expname in ["hmaml_scale", "hmaml_domain"]:
                            for info in data['result']:
                                if info['avg'] >= best_avg:
                                    best_avg = info['avg']
                                    best_info = info
                                    iterr = info['iteration']
                            data = best_info

                        if expname == "hmaml_scale":
                            print(f"Filtering on data size {szname}, {iterr} fname = {os.path.join(sz_dir,fname)}")
                        else:
                            print(f"Filtering on data size {szname}, fname = {os.path.join(sz_dir, fname)}")

                        for lang_id in meta_langs:
                            if curr.get(lang_id) is None:
                                curr[lang_id] = []
                            if data.get(lang_id) is not None:
                                cur_f1 = data[lang_id]["f1"]
                                curr[lang_id].append(cur_f1)

    # print(bdict)
    for expname in bdict:
        # print(f"{expname} --> {bdict[expname]}")
        print("\n################################")
        print(f"experiment name = {expname}")
        for mname in bdict[expname]:
            print(f"model name = {mname}")
            for szname in bdict[expname][mname]:
                print(f"sample size = {szname}")
                print("-----------------------")
                res_txt = ""
                all_f1s = []

                for key in bdict[expname][mname][szname]:
                    cur_f1s = bdict[expname][mname][szname][key]
                    all_f1s.extend(cur_f1s)
                    if cur_f1s:
                        mean, stdv = get_mean_stdv(cur_f1s)
                        res_txt += f"{key}: {mean:.3f}_{{{stdv:.3f}}}\t"
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


def baseline_report_modified(args):
    dir_path = "runs/finetune/hasoc2020"

    bdict = {}
    subdirs = os.listdir(dir_path)
    for target_lang in subdirs:
        subdirpath = os.path.join(dir_path, target_lang)
        subsubdirs = os.listdir(subdirpath)
        for fn in subsubdirs:
            model_name = fn.split("_")[0]
            fname = os.path.join(subdirpath, fn)
            if args.model_type not in model_name:
                continue
            if "_" + args.type not in fn:
                continue
            if args.samples and args.samples not in fn:
                continue
            if not args.samples and args.type + ".json" not in fn:
                continue
            print(f"Processing {fname}")
            if bdict.get(target_lang) is None:
                bdict[target_lang] = {}

            with open(fname, "r") as f:
                data = json.load(f)
            cur_f1 = data[args.type].get("f1")
            bdict[target_lang][model_name] = cur_f1

    print(json.dumps(bdict, indent=2))
    print(f"Reporting {dir_path}")
    for key in bdict:
        # print(f"{key} --> {bdict[key]}")
        res_txt = ""
        all_f1s = []
        for item in bdict[key]:
            all_f1s.append(bdict[key][item])
        if len(all_f1s) > 1:
            mean, stdv = get_mean_stdv(all_f1s)
            res_txt += f"{key}: {mean:.3f}_{{{stdv:.3f}}}"
        print(res_txt)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--exp_setting", required=True, type=str, help="The experimental scenario")
    parser.add_argument("--type", required=True, type=str, help="Tuning type")
    parser.add_argument("--model_type", required=True, default="bert", type=str, help="Model type")
    parser.add_argument("--epochs", required=True, type=str, help="Number of training epochs")
    parser.add_argument("--samples", default=None, type=str, help="Training set size")
    parser.add_argument("--shots", default=None, type=str, help="Number of support query")

    args = parser.parse_args()
    if args.exp_setting in ["hmaml", "xmetra", "xmaml", "fft"]:
        dir_path = "runs/summary/semeval2020/xmaml"
        meta_tuning_summary(args, dir_path)
    elif args.exp_setting == "finetune":
        baseline_report_modified(args)
    elif args.exp_setting == "scale":
        dir_path = "runs/summary/scale"
        scaling_summary_gen(dir_path)
    elif args.exp_setting == "domain":
        dir_path = "runs/summary/domain"
        scaling_summary_gen(dir_path)



if __name__ == "__main__":
    main()
