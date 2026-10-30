import json

import numpy as np

with open("results/gap_rule_confirmation/c1_replication.json", encoding="utf-8") as f:
    c1 = json.load(f)

print("=" * 110)
print("CRITERION C1: MODEL REPLICATION (STANDARD PROTOCOL)")
print("=" * 110)
print(
    f"{'Model':<25} | {'Verdict':<8} | {'d_FAR@80 (mean+-std, pos)':<28} | {'d_DIR@1 (mean+-std, pos)':<26} | {'d_DIR@5 (mean+-std, pos)':<26}"
)
print("-" * 110)
models = [
    "baseline_resnet18",
    "e0_baseline_val",
    "e1_cosface",
    "e1c_cosface_small",
    "e2_cosface_hardid",
    "e3_cosface_hardid_absneg",
    "e4_ce_hardid",
    "e5_ce_hardid_absneg",
]
for m in models:
    res = c1["models"][m]["lambda_0.5"]
    v = res["verdict"].upper()
    df = f"{res['d_far80_mean'] * 100:+5.2f} +- {res['d_far80_std'] * 100:4.2f} ({res['d_far80_pos']}/5)"
    dd1 = f"{res['d_dir1_mean'] * 100:+5.2f} +- {res['d_dir1_std'] * 100:4.2f} ({res['d_dir1_pos']}/5)"
    dd5 = f"{res['d_dir5_mean'] * 100:+5.2f} +- {res['d_dir5_std'] * 100:4.2f} ({res['d_dir5_pos']}/5)"
    print(f"{m:<25} | {v:<8} | {df:<28} | {dd1:<26} | {dd5:<26}")

print("\nSensitivity Analysis:")
print(
    f"{'Model':<25} | {'Verdict (l=0.25)':<16} | {'Verdict (l=0.50)':<16} | {'Verdict (l=0.75)':<16}"
)
print("-" * 80)
for m in models:
    v25 = c1["models"][m]["lambda_0.25"]["verdict"].upper()
    v50 = c1["models"][m]["lambda_0.5"]["verdict"].upper()
    v75 = c1["models"][m]["lambda_0.75"]["verdict"].upper()
    print(f"{m:<25} | {v25:<16} | {v50:<16} | {v75:<16}")

with open("results/gap_rule_confirmation/c2_cross_fitting.json", encoding="utf-8") as f:
    c2 = json.load(f)
print("\n" + "=" * 110)
print("CRITERION C2: CROSS-FITTING ON TEST IDENTITIES (e0_baseline_val)")
print("=" * 110)
for fold_id, fdata in c2["folds"].items():
    ch_l = fdata["chosen_lambda"]
    hcomp = fdata["heldout_comparison"]
    print(f"{fold_id}: {fdata['description']}")
    print(f"  Chosen lambda on tuning half: {ch_l}")
    print(
        f"  Held-out d_FAR@80: {hcomp['d_far80_mean'] * 100:+5.2f} +- {hcomp['d_far80_std'] * 100:4.2f}% ({hcomp['d_far80_pos']}/5 seeds positive)"
    )
    print(
        f"  Held-out d_DIR@5 : {hcomp['d_dir5_mean'] * 100:+5.2f} +- {hcomp['d_dir5_std'] * 100:4.2f}% ({hcomp['d_dir5_pos']}/5 seeds positive)"
    )
    print(
        f"  Held-out d_DIR@1 : {hcomp['d_dir1_mean'] * 100:+5.2f} +- {hcomp['d_dir1_std'] * 100:4.2f}% ({hcomp['d_dir1_pos']}/5 seeds positive)"
    )
    print(f"  Held-out positive: {fdata['heldout_positive']}")

with open("results/gap_rule_confirmation/c3_lookalike.json", encoding="utf-8") as f:
    c3 = json.load(f)
print("\n" + "=" * 110)
print("CRITERION C3: NO HARM ON LOOK-ALIKES")
print("=" * 110)
print(
    f"{'Model':<20} | {'Protocol':<15} | {'l':<4} | {'d_FAR@80 (mean+-std)':<22} | {'d_DIR@5 (mean+-std)':<22} | {'d_DIR@1 [noisy]':<20} | Safe"
)
print("-" * 110)
for m in ["e0_baseline_val", "baseline_resnet18"]:
    for proto in ["lookalike_p05", "lookalike_p01"]:
        for l_val in ["0.5", "0.25"]:
            d = c3["results"][m][proto][f"lambda_{l_val}"]
            comp = d["comparison"]
            df = f"{comp['d_far80_mean'] * 100:+5.2f} +- {comp['d_far80_std'] * 100:4.2f}%"
            dd5 = (
                f"{comp['d_dir5_mean'] * 100:+5.2f} +- {comp['d_dir5_std'] * 100:4.2f}%"
            )
            dd1 = (
                f"{comp['d_dir1_mean'] * 100:+5.2f} +- {comp['d_dir1_std'] * 100:4.2f}%"
            )
            safe = "YES" if d["no_harm_passed"] else "NO"
            print(
                f"{m:<20} | {proto:<15} | {l_val:<4} | {df:<22} | {dd5:<22} | {dd1:<20} | {safe}"
            )

with open("results/gap_rule_confirmation/gallery_sweep.json", encoding="utf-8") as f:
    sweep = json.load(f)
print("\n" + "=" * 110)
print("MECHANISM CHECK: GALLERY SIZE SWEEP")
print("=" * 110)
for m in ["e0_baseline_val", "baseline_resnet18"]:
    print(f"\nModel: {m}")
    print(
        f"{'Enrolled Size':<15} | {'Raw FAR@80':<12} | {'Raw DIR@5':<12} | {'d_FAR@80 (l=0.5)':<18} | {'d_DIR@5 (l=0.5)':<18} | {'d_DIR@1 (l=0.5)':<18}"
    )
    print("-" * 95)
    for s in ["25", "50", "100", "200", "full"]:
        sd = sweep[m][s]
        raw_runs = sd["raw"]
        raw_f80 = np.mean([r["far80"] for r in raw_runs]) * 100
        raw_d5 = np.mean([r["dir5"] for r in raw_runs]) * 100
        g50 = sd["lambda_0.5"]
        df50 = f"{g50['d_far80_mean'] * 100:+5.2f} +- {g50['d_far80_std'] * 100:4.2f}%"
        dd5_50 = f"{g50['d_dir5_mean'] * 100:+5.2f} +- {g50['d_dir5_std'] * 100:4.2f}%"
        dd1_50 = f"{g50['d_dir1_mean'] * 100:+5.2f} +- {g50['d_dir1_std'] * 100:4.2f}%"
        print(
            f"{s:<15} | {raw_f80:6.2f}%     | {raw_d5:6.2f}%     | {df50:<18} | {dd5_50:<18} | {dd1_50:<18}"
        )
