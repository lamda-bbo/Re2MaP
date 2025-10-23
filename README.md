# Re<sup>2</sup>MaP: <u>Ma</u>cro <u>P</u>lacement by <u>Re</u>cursively Prototyping and Packing Tree-based <u>Re</u>locating
This repository contains the code for Re<sup>2</sup>MaP, a macro placement framework that generates expert-quality macro layout recursively with novel and well-designed flow of prototyping and relocating.

We provide full implementation of our Re<sup>2</sup>MaP method.

## Run Re<sup>2</sup>MaP Algorithm

Below we provide detailed scripts to run the proposed Re<sup>2</sup>MaP algorithm.

### Build with Docker

We highly recommend the use of Docker to enable a smooth environment configuration.

The following steps are borrowed from [DREAMPlace](https://github.com/limbo018/DREAMPlace) repository. We make minor revisions to make it more clear.

1. Obtain the Docker image `limbo018/dreamplace:cuda`:

- Option 1: pull from the cloud [limbo018/dreamplace](https://hub.docker.com/r/limbo018/dreamplace).

  ```
  docker pull limbo018/dreamplace:cuda
  ```

- Option 2: build the image with Dockerfile from [DREAMPlace](https://github.com/limbo018/DREAMPlace.git).

  ```
  docker build . --file Dockerfile.DREAMPlace --tag limbo018/dreamplace:cuda
  ```

2. Build the Docker image for Re<sup>2</sup>MaP:

   ```
   docker build . --tag lamdabbo/re2map:cuda
   ```

3. Make sure you are in root directory of `Re2MaP` (e.g. `/path/to/Re2MaP`). Enter bash environment of the container.

   ```
   sudo docker run --gpus=all -it -v $(pwd):/workspace lamdabbo/re2map:cuda bash
   ```

4. Build. A shell script `build.sh` is offered, just run it in container.

   ```
   bash build.sh
   ```

### Get Benchmarks

In our experiments, we test our framework on cases from OpenROAD-flow-scripts (ORFS) [[Ajayi et al., DAC'19](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)], containing

- ariane133
- ariane136
- black_parrot
- bp_be
- bp_fe
- bp_multi
- swerv_wrapper

We run ORFS to generate the synthesized netlist and dump DEFs for our placement task. You can download the cases [here](https://drive.google.com/file/d/1AilCFLZIBDdvmsS2VWqO9ttqNIopNnTq/view?usp=sharing).

Then unzip the package and put it under following the directory:

```
Re2MaP/benchmarks/
```

### Run Macro Placement Task

You can run our experiment on all cases with shell script:

```
bash remap.sh all
```

Or, you can run single case by (ariane133 for example):

```
bash remap.sh ariane133
```

The macro placement results are stored at the following directory:

```
Re2MaP/install/results/${design_name}/${date}/${time}
```