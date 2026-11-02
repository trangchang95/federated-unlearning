"""Build data.json for the progress page from saved results; stops if a recomputed verdict disagrees."""
import json, re, statistics, subprocess, sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
def J(name, file="metrics.json"):
    return json.loads((repo / "results" / name / file).read_text())

D = {}
w1, w2, cn, cc = (J(x) for x in ["month1_week1", "month1_week2", "month1_week3_mnist_cnn", "month1_week3_cifar10_cnn"])
D["m1"] = {
    "baselines": [
        {"id": "lr", "acc": w1["accuracy"], "f1": w1["f1"], "dataset": "MNIST"},
        {"id": "mlp", "acc": w2["accuracy"], "f1": w2["f1"], "dataset": "MNIST"},
        {"id": "cnn", "acc": cn["accuracy"], "f1": cn["f1"], "dataset": "MNIST"},
        {"id": "cifar", "acc": cc["accuracy"], "f1": cc["f1"], "dataset": "CIFAR-10"},
    ],
    "curves": {k: [{"e": h["epoch"], "train": h["train_accuracy"], "val": h["validation_accuracy"]} for h in r["epoch_history"]]
               for k, r in {"mlp": w2, "cnn": cn, "cifar": cc}.items()},
    "perClass": {"cnn": cn["per_class_accuracy"], "cifar": cc["per_class_accuracy"]},
}

def w8(name):
    r = J(name); cfg_e = r["local_epochs"]
    return {
        "K": r["number_of_clients"], "E": cfg_e, "R": r["number_of_rounds"],
        "central": r["centralized"]["accuracy"], "fedavg": r["fedavg"]["accuracy"],
        "gap": r["accuracy_difference_fedavg_minus_centralized"],
        "bytes": r["fedavg"]["communication_cost_bytes"],
        "curveC": [{"x": h["epoch"], "y": h["validation_accuracy"]} for h in r["centralized"]["history"]],
        "curveF": [{"x": h["round"] * cfg_e, "y": h["validation_accuracy"]} for h in r["fedavg"]["history"]],
        "clients": [c["accuracy"] for c in r["fedavg"]["per_client_test_utility"]],
    }
D["m2"] = {"base": w8("month2_week8_mnist_iid_comparison"), "e2": w8("month2_week8_mnist_e2_variant"), "k10": w8("month2_week8_mnist_k10_variant")}

w9 = J("month3_week9_mnist_iid_vs_noniid")
central = D["m2"]["base"]["central"]
def part(rows): return [p["class_histogram"] for p in rows]
def setting(rows, res):
    return {"hist": part(rows), "acc": res["final_test_accuracy"], "f1": res["final_test_f1_macro"],
            "gap": res["final_test_accuracy"] - central,
            "curve": [{"x": h["round"], "y": h["test_accuracy_after_round"]} for h in res["history"]]}
S = {"iid": setting(w9["client_training_partitions"]["iid"], w9["iid"])}
for a in ["1.0", "0.5", "0.1"]:
    r = J(f"month3_week10_mnist_alpha{a}")
    S[a] = setting(r["client_training_partitions"], r["fedavg"])
S["shards"] = setting(w9["client_training_partitions"]["noniid"], w9["noniid"])
sw1, sw5 = J("month3_week11_mnist_fedprox_mu_sweep"), J("month3_week11_mnist_fedprox_mu_sweep_e5")
def sweep(r): return {"0": r["fedavg"]["final_test_accuracy"], **{m: r["fedprox"][m]["final_test_accuracy"] for m in ["0.01", "0.1", "1.0"]}}
bench = {}
for key, name in [("iid", "iid"), ("1.0", "alpha1.0"), ("0.5", "alpha0.5")]:
    bench[key] = {"fedavg": S[key]["acc"], "fedprox": J(f"month3_week12_{name}_fedprox")["fedprox"]["final_test_accuracy"]}
bench["0.1"] = {"fedavg": S["0.1"]["acc"], "fedprox": sw1["fedprox"]["0.01"]["final_test_accuracy"]}
assert sw1["fedavg"]["final_test_accuracy"] == S["0.1"]["acc"], "Week 11 FedAvg must reuse Week 10 alpha=0.1"
D["m3"] = {"settings": S, "sweep": {"e1": sweep(sw1), "e5": sweep(sw5)}, "bench": bench,
           "bytes": w9["noniid"]["communication_cost_bytes"]}

ROWS = ["global_test", "target_client", "non_target_clients", "target_client_dominant_classes", "target_client_shared_classes"]
def smoke(v):
    d = f"smoke_month4_week14_v3_mnist_{v}_federated_retrain"
    m, nf = J(d), J(d, "noise_floor.json")
    hist = part(m["client_training_partitions"])
    sizes = [sum(h) for h in hist]
    order = sorted(range(len(sizes)), key=lambda i: (sizes[i], i))
    t1 = order[(len(sizes) - 1) // 2]
    t2 = min(range(len(sizes)), key=lambda i: (-sizes[i], i))
    totals = [sum(h[c] for h in hist) for c in range(10)]
    shares = [(h[c] / totals[c] if totals[c] else 0.0, -i, c) for i, h in enumerate(hist) for c in range(10)]
    best = max(shares); t3, t3c = -best[1], best[2]
    assert t1 == nf["target_selection"]["chosen_client"], "T1 recount disagrees with the saved run"
    refs = {o["retrain_index"]: o["rows"] for o in nf["original_vs_retrains"]}
    floors = {}
    for key, name in [("A", "minibatch_order_only"), ("B", "also_initial_weights")]:
        f = nf["floors"][name]; rows = {}
        for row in ROWS:
            s = f["summary"][row]; n = s["examples"]
            if not n:
                rows[row] = {"n": 0}; continue
            ps = [p["rows"][row] if "rows" in p else p[row] for p in f["pairs"]]
            pairD = [abs(p["correct_x"] - p["correct_retrain"]) for p in ps]
            pairA = [p["agree"] for p in ps]
            assert abs(statistics.mean(pairD) - s["correct_difference_mean"]) < 1e-9
            assert abs(statistics.stdev(pairD) - s["correct_difference_std"]) < 1e-9
            limD = s["correct_difference_mean"] + 2 * s["correct_difference_std"]
            limA = s["agree_mean"] - 2 * s["agree_std"]
            mD, mA = [], []
            for i in f["retrain_indices"]:
                r = refs[i][row]; d_, a_ = abs(r["correct_x"] - r["correct_retrain"]), r["agree"]
                saved = f["original_within_floor_by_reference"][str(i)][row]
                assert (d_ <= limD) == saved["accuracy_gap_within"], (v, key, row, i, "gap flag")
                assert (a_ >= limA) == saved["agreement_within"], (v, key, row, i, "agreement flag")
                mD.append(d_); mA.append(a_)
            pc = f["positive_control_original_outside_floor_all_references"][row]
            assert pc["accuracy_gap"] == all(x > limD for x in mD)
            assert pc["agreement"] == all(x < limA for x in mA)
            rows[row] = {"n": n, "pairD": pairD, "pairA": pairA, "limD": limD, "limA": limA, "mD": mD, "mA": mA, "pcD": pc["accuracy_gap"], "pcA": pc["agreement"]}
        floors[key] = rows
    return {"sizes": sizes, "t1": t1, "t2": t2, "t3": t3, "t3class": t3c, "t3share": best[0],
            "dominant": nf["class_groups"]["target_dominant"], "floors": floors,
            "accM": refs[0]["global_test"]["accuracy_x"], "accR": refs[0]["global_test"]["accuracy_retrain"],
            "rounds": nf["config"]["number_of_rounds"], "train": nf["config"]["training_subset_size"]}
D["m4"] = {"a01": smoke("alpha0.1"), "iid": smoke("iid"),
           "week13Run": (repo / "results" / "month4_week13_mnist_full_retraining").exists()}
assert D["m4"]["week13Run"] is False, "Week 13 has now run: update the page text"
import csv, shutil
figdir = out.parent.parent / "figures"
figdir.mkdir(exist_ok=True)
FIGS = {
 "w1_digits": "month1_week1/sample_digits", "w1_cm": "month1_week1/confusion_matrix",
 "w2_curve": "month1_week2/learning_curve", "w2_cm": "month1_week2/confusion_matrix",
 **{f"w3_{k}_{n}": f"month1_week3_{d}/{m}" for k, d in [("cnn", "mnist_cnn"), ("cifar", "cifar10_cnn")] for n, m in [("curve", "learning_curve"), ("cm", "confusion_matrix"), ("maps", "feature_maps")]},
 **{f"w8_{k}_{n}": f"month2_week8_mnist_{d}/{m}" for k, d in [("base", "iid_comparison"), ("e2", "e2_variant"), ("k10", "k10_variant")] for n, m in [("conv", "convergence_comparison"), ("cmF", "fedavg_confusion_matrix"), ("cmC", "centralized_confusion_matrix")]},
 "w9_dist": "month3_week9_mnist_iid_vs_noniid/class_distribution", "w9_conv": "month3_week9_mnist_iid_vs_noniid/convergence_comparison",
 "w9_cmI": "month3_week9_mnist_iid_vs_noniid/iid_confusion_matrix", "w9_cmN": "month3_week9_mnist_iid_vs_noniid/noniid_confusion_matrix",
 **{f"w10_{a}_{n}": f"month3_week10_mnist_alpha{a}/{m}" for a in ["1.0", "0.5", "0.1"] for n, m in [("dist", "class_distribution"), ("conv", "convergence_comparison"), ("cm", "fedavg_confusion_matrix")]},
 "w11_e1": "month3_week11_mnist_fedprox_mu_sweep/convergence_comparison", "w11_e5": "month3_week11_mnist_fedprox_mu_sweep_e5/convergence_comparison",
}
for k, rel in FIGS.items():
    shutil.copyfile(repo / "results" / f"{rel}.png", figdir / f"{k}.png")
D["figSrc"] = {k: f"results/{rel}.png" for k, rel in FIGS.items()}
D["m1"]["classDist"] = list(csv.DictReader(open(repo / "results/month1_week1/class_distribution.csv")))
for k, d in [("base", "iid_comparison"), ("e2", "e2_variant"), ("k10", "k10_variant")]:
    r = J(f"month2_week8_mnist_{d}")
    D["m2"][k]["pcF"], D["m2"][k]["pcC"] = r["fedavg"]["per_class_accuracy"], r["centralized"]["per_class_accuracy"]
    D["m2"][k]["f1F"], D["m2"][k]["f1C"] = r["fedavg"]["f1_macro"], r["centralized"]["f1_macro"]
run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=repo, capture_output=True, text=True)
assert run.returncode == 0, "test suite failing"
D["tests"] = int(re.search(r"Ran (\d+) tests", run.stderr).group(1))
out.write_text(json.dumps(D, ensure_ascii=False, indent=1))
print("ok", {k: round(v["acc"], 4) for k, v in S.items()}, {k: v["fedprox"] for k, v in bench.items()})
for v in ("a01", "iid"):
    x = D["m4"][v]; print(v, x["t1"], x["t2"], x["t3"], x["t3class"], round(x["t3share"], 3), x["dominant"], round(x["accM"],3), round(x["accR"],3))
    for k in "AB":
        for r, c in x["floors"][k].items():
            if c["n"]: print(" ", k, r, c["n"], "limD", round(c["limD"],1), "mD", c["mD"], "pc", c["pcD"], "| limA", round(c["limA"],1), "mA", c["mA"], "pc", c["pcA"])
print({k:(v["K"],v["E"],v["R"],v["central"],v["fedavg"],round(v["gap"],4),v["bytes"]) for k,v in D["m2"].items()})
print(D["m3"]["sweep"], [b["acc"] for b in D["m1"]["baselines"]])
