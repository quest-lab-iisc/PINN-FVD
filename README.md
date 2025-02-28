# Finite-Volume-Discrete-Loss

# Project Title

Discrete Residual Loss Functions for Training Physics-Informed Neural Networks

# Project Description

 We propose a simulation-free PINNs training approach using a discrete numerical loss function instead of automatic differentiation, enabling efficient simulation of high Reynolds number flows. Our grid-based training strategy improves convergence, reduces computational cost, and outperforms traditional automatic differentiation. 

# Computational Time

The Kovasznay Flow experiments demonstrate that discrete loss requires less time compared to AD loss.

![Kovasznay Flow](Kovasznay%20Flow/Compare_AD_FVD.png)

# High Reynolds Number Flows

We have Shown results for Highly non-linear flows for Lid-Driven Cavity Problem

![Lid Driven Cavity](Lid%20Driven%20Cavity/Results/top_velocity_1_Re_1000_ICCS.png)

# Complex Geometry Cases

Finte Volume loss for complex geometries

![Flow Past Cylinder](Flow%20Past%20Cylinder/output/FPC.png)

 # Requirements

 ```sh
 pip install -r requirements.txt
 ```

 # Installation and Usage

```sh
git clone https://github.com/rishibabby/Finite-Volume-Discrete-Loss.git
cd Finite-Volume-Discrete-Loss/Kovasznay Flow
python KF_Vmodel_FVM.py
```
