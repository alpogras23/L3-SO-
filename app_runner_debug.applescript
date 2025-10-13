do shell script "bash -lc 'cd \"$HOME/Desktop/L3_SO_ANALYSIS\"; \
echo \"==== $(date) ====\" >> app_run.log; \
nohup ./Run_L3_SO.command >> app_run.log 2>&1 &'"
