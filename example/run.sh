rm -rf data
rm -rf figures
mkdir figures
mkdir data
cp ../src/LION/analysis_and_plots.py .
cp ../src/LION/clustering.py .
cp ../src/LION/data_collection.py .
python3 run.py
