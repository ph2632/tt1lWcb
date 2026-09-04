#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python

import os
import uproot


def print_branches(root_file, tree_name="Events", filter_keyword=None):
    print("=" * 120)
    print(f"[INFO] File: {root_file}")
    print(f"[INFO] Tree: {tree_name}")
    print("=" * 120)

    with uproot.open(root_file) as f:
        if tree_name not in f:
            print(f"[ERROR] Tree '{tree_name}' not found.")
            print("[INFO] Available keys:")
            for k in f.keys():
                print("  ", k)
            return

        tree = f[tree_name]
        branches = tree.keys()

        if filter_keyword is not None:
            branches = [b for b in branches if filter_keyword in b]

        print(f"[INFO] Number of branches: {len(branches)}")
        print("-" * 120)

        for b in branches:
            try:
                typename = tree[b].typename
            except Exception:
                typename = "Unknown"
            print(f"{b:<50} {typename}")

        print("=" * 120)


def check_required_branches(root_file, tree_name="Events"):
    required = [
        "xsecWeight",
        "genWeight",
        "lumiwgt",
        "puWeight",
        "trigEffWeight",
        "l1PreFiringWeight",

        "ak8_pt",
        "ak8_sdmass",
        "ak8_eta",
        "ak8_phi",
        "ak8_type",
        "ak8_n_c_in_jet",

        "score_cata_w_qq",
        "score_cata_qcd",
        "score_cata_top_bqq",
        "score_cata_top_bc",
        "score_cata_top_bq",
        "score_cata_non",

        "ak8_gpt_bc",
        "ak8_gpt_bb",
        "ak8_gpt_cc",
        "ak8_gpt_qcd",
        "ak8_gpt_bs",
        "ak8_gpt_qq",
        "ak8_gpt_cs",
        "ak8_gpt_topbw",
    ]

    with uproot.open(root_file) as f:
        tree = f[tree_name]
        branches = set(tree.keys())

    print("\n[CHECK] Required branch status")
    print("-" * 120)

    missing = []
    for b in required:
        ok = b in branches
        status = "OK" if ok else "MISSING"
        print(f"{b:<50} {status}")
        if not ok:
            missing.append(b)

    print("-" * 120)
    if len(missing) == 0:
        print("[SUMMARY] All required branches exist.")
    else:
        print(f"[SUMMARY] Missing {len(missing)} branches:")
        for b in missing:
            print("  -", b)


if __name__ == "__main__":
    base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
    mc_path = os.path.join(base_path, "MC/scored_samples_1merged_/")

    mc_files = [
        os.path.join(mc_path, f)
        for f in os.listdir(mc_path)
        if f.endswith(".root")
    ]

    if len(mc_files) == 0:
        raise RuntimeError(f"No ROOT files found in {mc_path}")

    test_file = mc_files[0]

    # 如果你想手动指定文件，取消下面这一行注释
    # test_file = "/path/to/your/file.root"

    print_branches(test_file, tree_name="Events")

    # 只看某类 branch，例如：
    # print_branches(test_file, tree_name="Events", filter_keyword="score")
    # print_branches(test_file, tree_name="Events", filter_keyword="ak8")

    check_required_branches(test_file, tree_name="Events")
