mkdir -p ./sample_data
cd ./sample_data
mkdir -p knot_theory
cd knot_theory
wget https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/test.csv
wget https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/train.csv
cd ..
mkdir m4_hourly_subset
cd m4_hourly_subset
wget https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv
wget https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv
cd ../..
