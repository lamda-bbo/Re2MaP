# usage: bash run.sh
orig=$(pwd)
cd build
make -j16
make -j16 install
cd ../install
designs=("ariane133" "ariane136" "black_parrot" "bp_be"  "bp_fe" "bp_multi" "swerv_wrapper" "bp_quad")
designs=("ariane133")
for design in "${designs[@]}"
do
    python dreamplace/Placer.py test/or_cases/$design.json
done
cd ${orig}
