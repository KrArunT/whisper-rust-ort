4️⃣ How to control it (runtime)
Default (best choice for EPYC)
```sh
NUMA_NODE=1 PIN_STRATEGY=ccd ./discover_optimal_model_epyc_ccd_fixed.sh
```

NUMA-wide (all cores in node)
```sh
PIN_STRATEGY=numa ./discover_optimal_model_epyc_ccd.sh
```

Disable pinning (debug)
```sh
PIN_STRATEGY=flat ./discover_optimal_model_epyc_ccd.sh
```
🧠 Why this matters for Whisper ONNX

Encoder + decoder are memory sensitive

CCD-local L3 = lower token latency

Removes noisy cross-CCD effects

Makes benchmark results repeatable
