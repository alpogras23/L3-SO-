# 🎯 PIPELINE REFERENCE CARD

## ✅ CURRENT STATUS
- **Step 1**: RUNNING (Preparing → Running soon)
- **Job ID**: `shy_roti_nkl78tcjgr`
- **Start Time**: 2025-12-06 22:36 UTC+3
- **Orchestrator PID**: 99869
- **All Files**: 240 AMOS22 (not just 15!)

---

## 📋 QUICK COMMANDS

### Monitor Real-time Log
```bash
tail -f /tmp/pipeline_log.txt
```

### Check Step 1 Status
```bash
cd /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml
source ../../.venv/bin/activate
az ml job show --name shy_roti_nkl78tcjgr --query status
```

### View Job Details
```bash
az ml job show --name shy_roti_nkl78tcjgr
```

### Stream Job Output
```bash
az ml job stream --name shy_roti_nkl78tcjgr
```

### View All Jobs
```
https://ml.azure.com/runs
```

### Check Orchestrator Status
```bash
ps aux | grep auto_pipeline_v3.sh
```

### View Full Log
```bash
cat /tmp/pipeline_log.txt
```

### Kill Orchestrator (if needed)
```bash
kill 99869
# Then restart if needed:
nohup bash /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml/auto_pipeline_v3.sh > /tmp/pipeline_log.txt 2>&1 &
```

---

## 📊 KEY JOB IDs

| Step | Job ID | Status |
|------|--------|--------|
| 1 | `shy_roti_nkl78tcjgr` | RUNNING |
| 2 | Auto-generated | Waiting for Step 1 |
| 3 | Auto-generated | Waiting for Step 2 |

---

## ⏱️ EXPECTED TIMELINE

| Time | Event |
|------|-------|
| Now | Step 1 running (Preparing) |
| +5-30min | Step 1 → Running |
| +2-3h | Step 1 → Completed, Step 2 starts (auto) |
| +4-7h | Step 2 → Completed, Step 3 starts (auto) |
| +7-8h | Step 3 → Completed ✅ |

---

## 📁 IMPORTANT PATHS

| Item | Path |
|------|------|
| Input Data | `/Users/alperenogras/Desktop/amos22/imagesTr/` |
| Manifest | `data/amos_manifest_full.csv` (240 entries) |
| Orchestrator Script | `.github/azure-ml/auto_pipeline_v3.sh` |
| Pipeline Status | `.github/azure-ml/PIPELINE_STATUS.md` |
| Real Teachers Script | `.github/azure-ml/step1_real_teachers_all.py` |
| Conda Env | `.github/azure-ml/conda.yaml` |
| Log File | `/tmp/pipeline_log.txt` |
| Job YAML | `.github/azure-ml/job_step1_teachers.yml` |

---

## 🎯 EXPECTED OUTPUTS

### Step 1 (TotalSegmentator)
```
azureml://jobs/shy_roti_nkl78tcjgr/outputs/teacher_labels/
├─ amos_0001/
│  ├─ hu_slice.png
│  ├─ psoas_left.png
│  ├─ psoas_right.png
│  ├─ vat.png
│  └─ metadata.json
├─ amos_0004/
│  └─ ... (same)
...
└─ amos_0600/
   └─ ... (240 cases total)
```

### Step 2 (U-Net Training)
```
azureml://jobs/[step2_job_id]/outputs/model/
└─ checkpoint.pth (trained weights)
```

### Step 3 (VFA/PMA Inference)
```
azureml://jobs/[step3_job_id]/outputs/results/
├─ metrics.json
├─ vfa_pma_report.csv
└─ overlays/ (visualization images)
```

---

## ⚠️ TROUBLESHOOTING

### If orchestrator stops
```bash
# Check if running
ps aux | grep auto_pipeline
# Restart if needed
nohup bash /Users/alperenogras/Desktop/L3_SO_ANALYSIS/.github/azure-ml/auto_pipeline_v3.sh > /tmp/pipeline_log.txt 2>&1 &
```

### If Step 1 fails
1. Check log: `tail -f /tmp/pipeline_log.txt`
2. Check Azure: `az ml job show --name shy_roti_nkl78tcjgr`
3. Check error: `az ml job stream --name shy_roti_nkl78tcjgr | tail -50`

### If network issues
- Azure ML jobs have built-in retry logic
- Orchestrator monitors with 30s polling interval
- Max wait: 10 hours for Step 1, 6 hours for Step 2, 30min for Step 3

---

## 📞 SUPPORT

| Issue | Solution |
|-------|----------|
| Log not updating | Check: `ls -lh /tmp/pipeline_log.txt` |
| Job not running | Check portal: https://ml.azure.com/runs |
| Need to restart | Kill PID 99869, manually restart script |
| Need to stop | Kill PID 99869, then `az ml job update --name shy_roti_nkl78tcjgr --set status=Canceled` |

---

## 📌 REMEMBER

✅ Pipeline is fully automated
✅ No manual intervention needed
✅ All 240 AMOS files being processed
✅ Orchestrator managing Steps 1 → 2 → 3
✅ Monitor with: `tail -f /tmp/pipeline_log.txt`

---

**Created**: 2025-12-06 22:36 UTC+3
**Last Updated**: Now
