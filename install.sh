if nvcc -V > /dev/null 2>&1; then
    echo "*********************************************************"
    echo "CUDA version found, installing requirements-dev-gpu.txt"
    echo "*********************************************************"
    echo ""
    pip install -r requirements-dev-gpu.txt
else
    echo "*********************************************************"
    echo "CUDA version NOT found, installing requirements-dev.txt"
    echo "*********************************************************"
    echo ""
    pip install -r requirements-dev.txt
fi