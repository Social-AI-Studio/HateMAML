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
        if args.exp_setting not in fname or args.type not in fname:
            continue
        if args.model_type not in model_name:
            continue
        with open(os.path.join(dir_path, fname)) as f:
            data = json.load(f)

        if args.type == "zeroshot":
            aux_lang = fname.split("_")[-2]
            target_lang = fname.split("_")[-1].strip(".json")
            epoch = fname.split("_")[2]
            shots = fname.split("_")[3]
            if args.samples is not None and args.type + "_" + args.samples not in fname:
                continue
            if args.samples is None and args.type + "_" + aux_lang not in fname:
                continue
            if epoch == args.epochs and shots == args.shots:
                print(f"Filtering on type = {args.type} epoch {epoch}, fname = {fname}")
                # print(json.dumps(data, indent=2))

                f1 = data[args.exp_setting].get("f1")
                key = aux_lang + "_" + target_lang
                if bdict.get(model_name) is None:
                    bdict[model_name] = {}
                bdict[model_name][key] = f1
        elif args.type == "fewshot":
            target_lang = fname.split("_")[-1].strip(".json")
            epoch = fname.split("_")[2]
            shots = fname.split("_")[3]
            if args.samples is not None and args.type + "_" + args.samples not in fname:
                continue
            if args.samples is None and args.type + "_" + target_lang not in fname:
                continue
            if epoch == args.epochs and shots == args.shots:
                print(f"Filtering on type = {args.exp_setting} epoch {epoch}, fname = {fname}")
                # print(json.dumps(data, indent=2))

                f1 = data[args.exp_setting].get("f1")
                key = target_lang
                if bdict.get(model_name) is None:
                    bdict[model_name] = {}
                bdict[model_name][key] = f1

        elif args.type == "zero-refine":
            target_lang = fname.split("_")[-1][:2]
            if "_" + str(args.shots) + "_" + target_lang not in str(fname):
                continue
            epoch = fname.split("_")[-3]
            if epoch == args.epochs:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["hmaml-zero-refine"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

    print(json.dumps(bdict, indent=2))
    print(dir_path)
    if args.type == "zeroshot":
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

                        if expname == "hmaml_scale":
                            print(f"Filtering on data size {szname}, fname = {os.path.join(sz_dir,fname)}")
                        else:
                            print(f"Filtering on data size {szname}, fname = {os.path.join(sz_dir, fname)}")

                        for lang_id in meta_langs:
                            if bdict[expname][mname][szname].get(lang_id) is None:
                                bdict[expname][mname][szname][lang_id] = []
                            cur_f1 = data[lang_id]["f1"]
                            bdict[expname][mname][szname][lang_id].append(cur_f1)

    for expname in bdict:
        # print(f"{expname} --> {bdict[expname]}")
        print(f"experiment name = {expname}")
        for mname in bdict[expname]:
            print(f"model name = {mname}")
            for szname in bdict[expname][mname]:
                print(f"sample size = {szname}")
                print("-----------------------")
                res_txt = ""
                all_f1s = []
                for key in bdict[expname][mname][szname]:
                    all_f1s.extend(bdict[expname][mname][szname][key])
                    mean, stdv = get_mean_stdv(bdict[expname][mname][szname][key])
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
    dir_path = "runs/finetune/semeval2020"

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
    if args.exp_setting in ["hmaml", "xmetra", "xmaml"]:
        dir_path = "runs/summary/semeval2020/hmaml_mixer_lit"
        meta_tuning_summary(args, dir_path)
    elif args.exp_setting == "finetune":
        baseline_report_modified(args)
    elif args.exp_setting == "scale":
        dir_path = "runs/summary/scale"
        scaling_summary_gen(dir_path)


if __name__ == "__main__":
    main()
