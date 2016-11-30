python setup.py build_ext --inplace || { echo "Build failed."; exit 1;}
find . -name '*.pyc' -delete

for test in $( find . -name "test*.py" ); do
  python -m $( echo $test | awk '{gsub("./stag", "stag"); gsub("/", "."); gsub(".py", ""); print}' ) || { echo "Test $test failed"; exit 1; };
done
