for playrecord in "$@"
do
    python -m retro.scripts.playback_movie ${playrecord} &
done
wait $(jobs -rp)
