design=$1
shift
bash build.sh
cd install
python scripts/remap_flow.py test/or_cases/${design}.json $@ 2>&1 | tee ${design}.log
