# Safety GAC
## Acknowledgement:
- [Openai Baselines](https://github.com/openai/baselines)

## Requirements
- python=3.5.2
- openai-gym=0.12.5 (mujoco200 is supported, but you need to use gym >= 0.12.5, it has a bug in the previous version.)
- mujoco-py=1.50.1.56 (~~**Please use this version, if you use mujoco200, you may failed in the FetchSlide-v1**~~)
- pytorch=1.0.0 (**If you use pytorch-0.4.1, you may have data type errors. I will fix it later.**)
- mpi4py
- joblib
- safety-gym

## TODO List
- [x] support GPU acceleration - although I have added GPU support, but I still not recommend if you don't have a powerful machine.
- [x] add multi-env per MPI.
- [x] add the plot and demo of the **FetchSlide-v1**.

## Instruction to run the code
```bash
(nohup mpirun -np 1 python -u train.py --env-name='Safexp-PointGoal0-v0' --cuda --alg gac --n-epochs=10 > ./logs/Safexp-PointGoal0-v0-gac.log 2>&1 &)
```

