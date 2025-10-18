if [[ $# -gt 0 ]]; then
    option=$1
    shift
else
    echo "usage:"
    echo "  bash remap.sh <design>"
    echo "  bash remap.sh all"
    echo "  bash remap.sh exp <experiment>"
    exit
fi
bash build.sh
cd install
mkdir -p logs
case "$option" in 
    all)
        all_designs=(ariane133 ariane136 bp bp_be bp_fe bp_multi swerv_wrapper)
        echo $all_designs
        for design in ${all_designs[*]}
        do
            command=(python scripts/remap_flow.py test/or_cases/${design}.json $@)
            echo "RUN ${command[*]} 2>&1 | tee logs/${design}.log"
            "${command[@]}" 2>&1 | tee logs/${design}.log
        done
    ;;
    exp)
        experiment=$1
        shift
        command=(python scripts/remap_flow.py ${experiment}.json $@)
        echo "RUN ${command[*]} 2>&1 | tee logs/${experiment}.log"
        "${command[@]}" 2>&1 | tee logs/${experiment}.log
    ;;
    *)
        design=$option
        command=(python scripts/remap_flow.py test/or_cases/${design}.json $@)
        echo "RUN ${command[*]} 2>&1 | tee logs/${design}.log"
        "${command[@]}" 2>&1 | tee logs/${design}.log
    ;;
esac