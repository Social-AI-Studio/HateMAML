import os
import json
import statistics


def get_mean_stdv(data: list):
    mean = statistics.mean(data)
    stdv = statistics.stdev(data)
    print(f"mean {mean} stdv {stdv}")


def read_files(dir_path: str):
    files = os.listdir(dir_path)
    bdict = {}
    identifier = "fewshot"
    model_type = "bert"
    few_flag = False

    if "semeval" in dir_path:
        langs = ["ar", "tr", "gr", "da"]
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
            if few_flag and "full" not in fname:
                continue

            if few_flag:
                epoch = int(fname.split("_")[-5])
            else:
                epoch = int(fname.split("_")[-4])

            if epoch == 5:
                print(f"Filtering on epoch {epoch}, fname = {fname}")
                type = "hmaml-zeroshot"
                if few_flag:
                    type = "full"

                f1 = data[type].get("f1")

                key = aux_lang + "_" + target_lang
                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][key] = f1
        elif identifier == "zero-refine" and identifier in str(fname):
            target_lang = fname.split("_")[-1][:2]
            epoch = int(fname.split("_")[-3])
            if epoch == 9:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["hmaml-zero-refine"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

        elif identifier == "_maml" and identifier in str(fname):
            target_lang = fname.split("_")[-1][:2]
            epoch = int(fname.split("_")[-3])
            if epoch == 5:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["maml"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

        elif identifier == "fewshot" and identifier in str(fname):
            target_lang = fname.split("_")[-1][:2]
            epoch = int(fname.split("_")[-3])
            if epoch == 9:
                print(f"Filtering on epoch {epoch}, fname = {fname}")

                f1 = data["hmaml-fewshot"].get("f1")

                if bdict.get(model_name) is None:
                    bdict[model_name] = {}

                bdict[model_name][target_lang] = f1

    print(json.dumps(bdict, indent=2))

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
                    get_mean_stdv(f1_list)
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
                get_mean_stdv(f1_list)


def main():
    dir_path = "runs/summary/hateval2019/hmaml_vanilla_lit"
    read_files(dir_path)


if __name__ == '__main__':
    main()
