# Neural Monte Carlo Fluid Simulation
This is the official code for Neural Monte Carlo Fluid Simulation by Pranav Jain, Ziyin Qu, Peter Yichen Chen, Oded Stein

<p align="center">
  <img src="final_material/karman/karman.gif"/>
  <br>
  <em>Karman Vorticity</em>
</p>
  
<p align="center">
  <img src="final_material/smoke_obstacle/smoke_obstacle.gif" height="25%" width="25%"/>
  <img src="final_material/vortex_collide/vortex_collide.gif" height="25%" width="25%"/>
  <br>
  <em>Density of smoke with obstacle&nbsp</em>
  <em>&nbspDensity of smoke rings colliding</em>
</p>

## Setup
This code is tested on _x64 linux_ platform using _Python 3.10.12_.
First clone the repository using
```sh
git clone --recurse-submodules https://github.com/Pranav-Jain/fluid_insr_wost.git
```
Install all packages from requirements.txt into a clean Python environment. Use the following command to do that:
```sh
pip install -r requirements.txt
```
For 3D examples, pyopenvdb needs to be installed separately to save vdb files for visualization. 
To install, pyopenvdb, it requires the following dependencies: boost, tbb, c-blosc, pybind11
```sh
apt-get install -y libboost-iostreams-dev
apt-get install -y libtbb-dev
apt-get install -y libblosc-dev

cd lib/pybind11
mkdir build
cd build
cmake ..
make install
```
Once the dependencies are installed, pyopenvdb can be installed using
```sh
cd lib/openvdb
mkdir build
cd build
cmake .. -D OPENVDB_BUILD_PYTHON_MODULE=ON -D USE_NUMPY=ON
make install
```

You would have to build your own zombie Python bindings and move them to __src/2d__ or __src/3d__ respectively. For 2d, do the following
```sh
cd bindings/zombie/
mkdir build
cd build
cmake ..
make -j4
```
This should create the bindings in the build directory. Move the bindings to __src/2d__.
Bindings for 3D can be built similarly by going into __bindings/zombie3d__. Make sure to move the bindings to __src/3d__.

## Run
Once you have everything setup, go to the examples directory and choose the example you want and use the file __run.sh__ to run. For example:
```sh
cd examples/karman
bash run.sh
```
A new directory __results/__ will be created which wil store the output.

## Testing
You could test with different parameters by changing their values in __run.sh__.

## Computing Error with Other Methods
Go to the __experiments__ directory and choose the algorithm you want and use the file __run.sh__ to run. For example
```sh
cd experimets/pinnFluid
bash run.sh
```
