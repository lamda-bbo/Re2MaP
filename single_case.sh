# usage: bash single_case.sh ariane133
orig=$(pwd)
cd build
make -j16
make -j16 install
cd ../install
for design in $@
do
    python dreamplace/Placer.py test/or_cases/$design.json
done
cd ${orig}
