# ADR 0001: Assumed Defaults

Status: Accepted on 2026-10-24.

## Context

The project must produce a tested repository, a local demo, and a cluster
package within one week.

## Decisions

1. The first target is complete code, a synthetic demo, and a cluster-ready
   package. Real training is the second gate.
2. ADT is the target quantitative dataset in version 1.
3. The oracle protocol uses ground-truth masks, depth, and poses.
4. SAM 2 masks are the first realistic extension.
5. The scheduler is Slurm.
6. The container runtime is Apptainer.
7. The cluster values stay as placeholders until the user supplies them.
8. The user runs all cluster jobs.
9. The repository is private.
10. The code licence is MIT.
11. The cluster gives about 10 hours on H100 and H200 GPUs.
12. The cluster has Slurm and internet access.
13. The user runs all cluster jobs.
14. The GitHub account is `jagerknight10`.
15. The repository is private and named `wise-egomap`.
16. ADT is blocked until the licence is approved.

## Open Items

1. Request the ADT licence.
2. Confirm the cluster partition, account, and container runtime.
3. Confirm whether real room capture is possible.
4. Confirm who inspects the 200 sampled label windows.
