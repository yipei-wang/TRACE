# Overview
This repository is the source code to implement TRAjectory importanCE (TRACE), a benchmark for deletion tests in eXplainable Artificial Intelligence (XAI). 

**TRACE** is a framework to solve for the best trajectories under the popular deletion test in XAI evaluations. By achieving the best possible results, TRACE can be used as the ground truth of the metric and is also the most principled explanation under deletions.

In the deletion test, features are deleted gradually, and the AUCs of the corresponding curves are used as the measurement. Take GradCAM as an example:
![alt text](https://github.com/yipei-wang/Images/blob/main/TRACE/TRACE_demonstration.png)

TRACE achieves the best score under such metrics. A deletion process (with patch mean) is shown as follows:
![alt text](https://github.com/yipei-wang/Images/blob/main/TRACE/TRACE-Greedy-Le_image1.png)

# Contents

**Libraries**
For the implementation of TRACE, please have
<pre>
  numpy==1.19.5
  torch==1.10.2
  torchvision=0.11.3
</pre>
In order to carry out the comparison, please also have
<pre>
  captum==0.5.0
  torchray==1.0.0.2
</pre>

**Tutorials**
For the results of TRACE-Greedy, please run
<pre>
  python Greedy.py 
</pre>
This will generate results of TRACE-Greedy-Le and TRACE-Greedy-Mo, and plot the deletion process and the deletion scores.

For the comparison between TRACE-Greedy and attribution methods, please run
<pre>
  python Greedy.py --compare
</pre>
This will also include attribution explanation methods such as IG, GradCAM, etc. in the deletion score figure.

Similarly, please run the following commands for the TRACE-SA results and comparisons
<pre>
  python SimulatedAnnealing.py
  python SimulatedAnnealing.py --compare
</pre>
