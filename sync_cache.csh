#!/bin/tcsh -f
# sync_cache.csh -- keep a fast node-local copy of the derived parquet cache.
#
#   EOS-fuse random reads of the big parquets (tt-semi ~11 GB) are slow;
#   /tmp on lxplus is node-local disk (/dev/vdb, ~90 GB free).  Copy the
#   cache there once per node and point WCB_CACHE_DIR at it.
#
# Usage:
#   ./sync_cache.csh pull            # EOS  -> local   (before a run)
#   ./sync_cache.csh push            # local -> EOS    (persist new builds)
#   ./sync_cache.csh pull v18_gptnodes   # a different cache_tag substring
#
# Then, in the shell you run Make_plots.py from:
#   setenv WCB_CACHE_DIR /tmp/$USER/wcb_cache_local/
#
# For the one-time v20 re-derive: set WCB_CACHE_DIR to the LOCAL dir first so
# the new parquets are WRITTEN to fast local disk, then `push` them to EOS.
#
# CAVEAT: /tmp is per-node and wiped on node switch / reboot.  Anything a
# run rebuilds lands in the LOCAL dir -- run `./sync_cache.csh push` before
# logging out so EOS keeps the authoritative copy.

set EOS   = /eos/user/a/agapitos/wcb_cache
set LOCAL = /tmp/$USER/wcb_cache_local
set TAG   = derived_v20_bake_v1
if ( $#argv >= 2 ) set TAG = "$argv[2]"

set MODE = "$argv[1]"
if ( "$MODE" != "pull" && "$MODE" != "push" ) then
    echo "usage: $0 {pull|push} [cache_tag_substring]"
    exit 1
endif

mkdir -p $LOCAL

if ( "$MODE" == "pull" ) then
    echo "[sync] $EOS  ->  $LOCAL   (tag: *$TAG*)"
    rsync -a --info=progress2 --update \
        $EOS/*__${TAG}.parquet $LOCAL/
else
    echo "[sync] $LOCAL  ->  $EOS   (tag: *$TAG*)"
    rsync -a --info=progress2 --update \
        $LOCAL/*__${TAG}.parquet $EOS/
endif

echo "[sync] done.  local dir:"
ls -lh $LOCAL/*__${TAG}.parquet | awk '{print "  "$5"\t"$9}'
echo "[sync] set:  setenv WCB_CACHE_DIR $LOCAL/"
