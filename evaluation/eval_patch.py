import os
ROOT = os.environ.get("KCHANNEL_ROOT", os.path.expanduser("~/k-channel-data"))
import sys
which = sys.argv[1]; sys.argv = sys.argv[:1]
V = '[("k-Channel v7 k5", "genrelease/k-5_microagg_v7"), ("k-Channel v7 k10", "genrelease/k-10_microagg_v7"), ("k-Channel v7 k25", "genrelease/k-25_microagg_v7")]'
if which == "attr":
    src = open(ROOT + "/logs/attr_fidelity.py").read(); i, j = src.index("RUNS = {"), src.index("for ds, runs in RUNS.items():")
    src = src[:i] + 'RUNS = {"celeba": ' + V + ', "mtf": ' + V + '}\n' + src[j:]
    assert src.count('/ "attr_fidelity.csv"') == 1; src = src.replace('/ "attr_fidelity.csv"', '/ "attr_fidelity_v7.csv"')
else:
    src = open(ROOT + "/logs/u40.py").read(); i, j = src.index("RUNS = ["), src.index("def release(d):")
    src = src[:i] + "RUNS = " + V + "\n" + src[j:]
    assert src.count('OUT / "u40.csv"') == 2; src = src.replace('OUT / "u40.csv"', 'OUT / "u40_v7.csv"')
exec(compile(src, which + "_v7", "exec"))
