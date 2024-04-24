git pull
scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 os=linux arch=armv8a -j4
cp build/libarm_compute* /home/run/lib 
cp build/examples/graph_vanilla_transformer /home/run 